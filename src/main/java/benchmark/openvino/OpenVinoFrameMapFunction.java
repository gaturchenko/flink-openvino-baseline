package benchmark.openvino;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.functions.RichMapFunction;

public final class OpenVinoFrameMapFunction
        extends RichMapFunction<PreprocessedFrameRecord, DetectionResult> {

    private final String modelPath;
    private final String device;
    private final int numStreams;
    private final int numThreads;
    private final String statsCsvPath;
    private final String runId;

    private transient long handle;
    private transient float[] output;
    private transient int expectedInputSize;
    private transient LatencyStats stats;
    private transient int subtaskIndex;
    private transient int parallelSubtasks;

    public OpenVinoFrameMapFunction(
            String modelPath,
            String device,
            int numStreams,
            int numThreads,
            String statsCsvPath,
            String runId) {
        this.modelPath = modelPath;
        this.device = device;
        this.numStreams = numStreams;
        this.numThreads = numThreads;
        this.statsCsvPath = statsCsvPath;
        this.runId = runId;
    }

    @Override
    public void open(OpenContext openContext) {
        subtaskIndex = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
        parallelSubtasks = getRuntimeContext().getTaskInfo().getNumberOfParallelSubtasks();
        stats = new LatencyStats();

        handle = NativeOpenVinoBridge.createEngine(
                modelPath,
                device,
                numStreams,
                numThreads
        );

        int inputSize = NativeOpenVinoBridge.inputSize(handle);
        int outputSize = NativeOpenVinoBridge.outputSize(handle);

        expectedInputSize = inputSize;
        output = new float[outputSize];

        System.out.println(
                "Initialized OpenVINO frame engine. "
                        + "inputSize=" + inputSize
                        + ", outputSize=" + outputSize
                        + ", numStreams=" + numStreams
                        + ", numThreads=" + numThreads
        );
    }

    @Override
    public DetectionResult map(PreprocessedFrameRecord frame) {
        long startNs = System.nanoTime();

        if (frame.inputTensor.length != expectedInputSize) {
            throw new IllegalArgumentException(
                    "Unexpected preprocessed input size. Expected "
                            + expectedInputSize
                            + ", but got "
                            + frame.inputTensor.length
            );
        }

        int rc = NativeOpenVinoBridge.predict(handle, frame.inputTensor, output);

        if (rc != 0) {
            throw new RuntimeException("OpenVINO predict failed with rc=" + rc);
        }

        float maxScore = max(output);
        long endNs = System.nanoTime();
        stats.record(startNs, endNs);

        return new DetectionResult(maxScore, frame.e2eStartNs);
    }

    @Override
    public void close() throws Exception {
        try {
            if (stats != null) {
                StatsCsvWriter.record("inference", stats);
            }
        } finally {
            if (handle != 0) {
                NativeOpenVinoBridge.destroyEngine(handle);
                handle = 0;
            }
        }
    }

    private static float max(float[] values) {
        float m = -Float.MAX_VALUE;

        for (float v : values) {
            if (v > m) {
                m = v;
            }
        }

        return m;
    }
}
