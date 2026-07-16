package benchmark.openvino;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Accumulates per-scope latency stats during a run and, once the job finishes, emits ONE wide row
 * per run whose columns line up with the XXXX-2 / TorchServe comparison tables produced by
 * process_results.py / process_torchserve_results.py. That way a Flink baseline row drops straight
 * into the OSU-RGB framework comparison, substituting the XXXX-2-UDF case.
 *
 * Two scopes are recorded: "e2e" (parse+preprocess+infer, from E2eStatsMapFunction) and
 * "inference" (pure OpenVINO predict, from OpenVinoFrameMapFunction). Subtask stats are merged, so
 * the row is the aggregate across Flink parallelism. Call configureRun() before the job and
 * writeWideRow() after env.execute() returns.
 */
public final class StatsCsvWriter {
    private static final Object LOCK = new Object();

    /** scope ("e2e" | "inference") -> stats merged across parallel subtasks for the current run. */
    private static final Map<String, LatencyStats> SCOPES = new HashMap<>();

    // Run identity, matching the XXXX-2/TorchServe comparison columns. Set by the job; safe defaults.
    private static volatile String queryName = "flink";
    private static volatile String paramName = "baseline";
    private static volatile String paramValue = "flink";
    private static volatile String repetition = "0";

    private static final String HEADER = String.join(",",
            "query_name",
            "inference_config_param_name",
            "inference_config_param_value",
            "repetition",
            "flink_records",
            "end_to_end_duration_us",    // wall-clock span first-arrival -> last-output, cf. XXXX-2
            "end_to_end_throughput",     // records/s, cf. XXXX-2 end_to_end_throughput
            "end_to_end_latency_us",     // mean e2e latency, cf. XXXX-2 end_to_end_latency_us
            "e2e_latency_ms_mean",       // cf. TorchServe e2e_latency_ms_*
            "e2e_latency_ms_p50",
            "e2e_latency_ms_p95",
            "e2e_latency_ms_p99",
            "inference_latency_ms_mean", // pure-inference distribution (Flink-specific detail)
            "inference_latency_ms_p50",
            "inference_latency_ms_p95",
            "inference_latency_ms_p99"
    );

    private StatsCsvWriter() {}

    /** Set the comparison-key columns (query/config/repetition) before the job runs. */
    public static void configureRun(String query, String pName, String pValue, String rep) {
        if (query != null && !query.isBlank()) {
            queryName = query;
        }
        if (pName != null && !pName.isBlank()) {
            paramName = pName;
        }
        if (pValue != null && !pValue.isBlank()) {
            paramValue = pValue;
        }
        if (rep != null && !rep.isBlank()) {
            repetition = rep;
        }
    }

    /** Buffer one subtask's stats for a scope, merging across parallel subtasks. */
    public static void record(String scope, LatencyStats stats) {
        if (scope == null || stats == null) {
            return;
        }
        synchronized (LOCK) {
            SCOPES.computeIfAbsent(scope, ignored -> new LatencyStats()).merge(stats);
        }
    }

    /** Write the single wide row for this run. Call once, after env.execute() returns. */
    public static void writeWideRow(String statsCsvPath) throws IOException {
        if (statsCsvPath == null || statsCsvPath.isBlank()) {
            return;
        }

        Path path = Path.of(statsCsvPath);
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        synchronized (LOCK) {
            LatencyStats e2e = SCOPES.get("e2e");
            LatencyStats inference = SCOPES.get("inference");

            boolean writeHeader = !Files.exists(path) || Files.size(path) == 0L;
            try (BufferedWriter writer = Files.newBufferedWriter(
                    path,
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND
            )) {
                if (writeHeader) {
                    writer.write(HEADER);
                    writer.newLine();
                }
                writer.write(formatRow(e2e, inference));
                writer.newLine();
            }
            SCOPES.clear();
        }
    }

    private static String formatRow(LatencyStats e2e, LatencyStats inference) {
        long records = e2e != null ? e2e.count() : (inference != null ? inference.count() : 0L);
        double e2eDurationUs = e2e != null ? e2e.elapsedNs() / 1_000.0 : 0.0;
        double e2eThroughput = e2e != null ? e2e.throughputRecordsPerSecond() : 0.0;
        double e2eLatencyUs = e2e != null ? e2e.averageLatencyNs() / 1_000.0 : 0.0;

        return String.join(",",
                csv(queryName),
                csv(paramName),
                csv(paramValue),
                csv(repetition),
                Long.toString(records),
                formatDouble(e2eDurationUs),
                formatDouble(e2eThroughput),
                formatDouble(e2eLatencyUs),
                msMean(e2e),
                msPercentile(e2e, 50.0),
                msPercentile(e2e, 95.0),
                msPercentile(e2e, 99.0),
                msMean(inference),
                msPercentile(inference, 50.0),
                msPercentile(inference, 95.0),
                msPercentile(inference, 99.0)
        );
    }

    private static String msMean(LatencyStats stats) {
        return formatDouble(stats == null ? 0.0 : stats.averageLatencyNs() / 1_000_000.0);
    }

    private static String msPercentile(LatencyStats stats, double percentile) {
        return formatDouble(stats == null ? 0.0 : stats.percentileLatencyNs(percentile) / 1_000_000.0);
    }

    private static String formatDouble(double value) {
        return String.format(Locale.ROOT, "%.3f", value);
    }

    private static String csv(String value) {
        String safeValue = value == null ? "" : value;
        if (!safeValue.contains(",") && !safeValue.contains("\"") && !safeValue.contains("\n")) {
            return safeValue;
        }
        return "\"" + safeValue.replace("\"", "\"\"") + "\"";
    }
}
