package benchmark.openvino;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.util.Collector;

import java.io.StringReader;

/**
 * Parses the single base64 column out of each CSV line and stamps the record's ingestion time.
 *
 * To keep the framework comparison fair, ingestion is paced at a fixed rate matching the XXXX-2 / TorchServe
 * TCP sources (MOCK_TUPLE_RATE): without this, Flink's FileSource is a backpressure-driven firehose while
 * the XXXX-2/TorchServe sources emit at a controlled tuples/second. The pacing is cumulative (release the
 * i-th record no earlier than start + i*interval), mirroring the XXXX-2 TCPDataServer SendRateLimiter, and
 * {@code e2eStartNs} is stamped AFTER pacing so it is the true (paced) arrival time -- the same point XXXX-2
 * stamps its ingestion timestamp. With parallelism > 1 the rate is split across subtasks so the aggregate
 * ingestion rate matches the single XXXX-2 source.
 */
public final class CsvBase64OnlyParser extends RichFlatMapFunction<String, FrameRecord> {
    private final int base64Column;
    private final boolean hasHeader;
    private final double ratePerSecond;   // <= 0 => unbounded (no pacing)

    private transient boolean firstLine;
    private transient long intervalNs;    // per-subtask inter-record interval; 0 => unbounded
    private transient long startNs;
    private transient long emitted;

    public CsvBase64OnlyParser(int base64Column, boolean hasHeader, double ratePerSecond) {
        this.base64Column = base64Column;
        this.hasHeader = hasHeader;
        this.ratePerSecond = ratePerSecond;
    }

    @Override
    public void open(OpenContext openContext) {
        firstLine = true;
        emitted = 0L;
        startNs = 0L;
        int subtasks = getRuntimeContext().getTaskInfo().getNumberOfParallelSubtasks();
        double perSubtask = ratePerSecond > 0 ? ratePerSecond / Math.max(1, subtasks) : 0.0;
        intervalNs = perSubtask > 0 ? (long) (1_000_000_000.0 / perSubtask) : 0L;
    }

    @Override
    public void flatMap(String line, Collector<FrameRecord> out) throws Exception {
        if (line == null || line.isBlank()) {
            return;
        }

        if (hasHeader && firstLine) {
            firstLine = false;
            return;
        }
        firstLine = false;

        // Fixed-rate ingestion pacing (see class doc). Sleep until this record's scheduled slot,
        // then stamp arrival, so latency is measured against a controlled arrival rate.
        if (intervalNs > 0L) {
            if (emitted == 0L) {
                startNs = System.nanoTime();
            }
            long sleepNs = (startNs + emitted * intervalNs) - System.nanoTime();
            if (sleepNs > 0L) {
                Thread.sleep(sleepNs / 1_000_000L, (int) (sleepNs % 1_000_000L));
            }
        }
        emitted++;

        long e2eStartNs = System.nanoTime();

        CSVParser parser = CSVFormat.DEFAULT
                .builder()
                .setTrim(true)
                .build()
                .parse(new StringReader(line));

        for (CSVRecord record : parser) {
            out.collect(new FrameRecord(record.get(base64Column), e2eStartNs));
        }
    }
}
