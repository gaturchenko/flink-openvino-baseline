package benchmark.openvino;

public final class NativeOpenVinoBridge {
    static {
        System.loadLibrary("flink_openvino_bridge_jni");
    }

    private NativeOpenVinoBridge() {}

    public static native long createEngine(
            String modelPath,
            String device,
            int numStreams,
            int numThreads
    );

    public static native int inputSize(long handle);

    public static native int outputSize(long handle);

    public static native int predict(
            long handle,
            float[] input,
            float[] output
    );

    public static native void destroyEngine(long handle);
}
