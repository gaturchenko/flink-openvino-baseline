package benchmark.openvino;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.connector.file.src.FileSource;
import org.apache.flink.connector.file.src.reader.TextLineInputFormat;
import org.apache.flink.core.fs.Path;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public final class OpenVinoCsvObjectDetectionJob {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println(
                    "Usage: OpenVinoCsvObjectDetectionJob "
                            + "<modelPath> <csvPath> "
                            + "[device] [flinkParallelism] [ovStreams] [ovThreads] "
                            + "[base64Column] [hasHeader] [width] [height] [printResults] "
                            + "[statsCsvPath]"
            );
            System.exit(1);
        }

        String modelPath = args[0];
        String csvPath = args[1];

        String device = args.length >= 3 ? args[2] : "CPU";
        int flinkParallelism = args.length >= 4 ? Integer.parseInt(args[3]) : 1;
        int ovStreams = args.length >= 5 ? Integer.parseInt(args[4]) : 1;
        int ovThreads = args.length >= 6 ? Integer.parseInt(args[5]) : 1;

        int base64Column = args.length >= 7 ? Integer.parseInt(args[6]) : 1;
        boolean hasHeader = args.length >= 8 && Boolean.parseBoolean(args[7]);

        int width = args.length >= 9 ? Integer.parseInt(args[8]) : 320;
        int height = args.length >= 10 ? Integer.parseInt(args[9]) : 320;

        boolean printResults = args.length >= 11 && Boolean.parseBoolean(args[10]);
        String statsCsvPath = args.length >= 12 ? args[11] : "target/openvino-object-detection-stats.csv";
        String runId = "run-" + System.currentTimeMillis();

        // Optional comparison-key columns so the emitted row slots into the XXXX-2/TorchServe table.
        String queryName = args.length >= 13 ? args[12] : "flink_osu_rgb";
        String paramName = args.length >= 14 ? args[13] : "baseline";
        String paramValue = args.length >= 15 ? args[14] : "flink";
        String repetition = args.length >= 16 ? args[15] : "0";
        StatsCsvWriter.configureRun(queryName, paramName, paramValue, repetition);

        // Fixed ingestion rate (tuples/s) to match the XXXX-2/TorchServe TCP source MOCK_TUPLE_RATE.
        // <= 0 means unbounded (backpressure-paced); pass the same value as MOCK_TUPLE_RATE for parity.
        double ratePerSecond = args.length >= 17 ? Double.parseDouble(args[16]) : 0.0;

        StreamExecutionEnvironment env =
                StreamExecutionEnvironment.getExecutionEnvironment();

        env.setParallelism(flinkParallelism);

        FileSource<String> source = FileSource
                .forRecordStreamFormat(
                        new TextLineInputFormat(),
                        new Path(csvPath)
                )
                .build();

        var results = env
                .fromSource(
                        source,
                        WatermarkStrategy.noWatermarks(),
                        "csv-base64-frame-source"
                )
                .flatMap(new CsvBase64OnlyParser(base64Column, hasHeader, ratePerSecond))
                .name("parse-base64-frames")
                .setParallelism(1)
                .map(new FramePreprocessMapFunction(width, height))
                .name("preprocess-frames")
                .map(new OpenVinoFrameMapFunction(
                        modelPath,
                        device,
                        ovStreams,
                        ovThreads,
                        statsCsvPath,
                        runId
                ))
                .name("openvino-object-detection")
                .map(new E2eStatsMapFunction(statsCsvPath, runId))
                .name("collect-e2e-stats");

        if (printResults) {
            results.print();
        } else {
            results.sinkTo(new DiscardSink<>());
        }

        env.execute("Flink OpenVINO CSV Object Detection");

        // Operators have closed and buffered their scope stats; emit the single wide comparison row.
        StatsCsvWriter.writeWideRow(statsCsvPath);
    }
}
