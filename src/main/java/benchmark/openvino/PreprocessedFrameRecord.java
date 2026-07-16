package benchmark.openvino;

import java.io.Serializable;
import java.util.Objects;

public final class PreprocessedFrameRecord implements Serializable {
    public final float[] inputTensor;
    public final long e2eStartNs;

    public PreprocessedFrameRecord(float[] inputTensor, long e2eStartNs) {
        this.inputTensor = Objects.requireNonNull(inputTensor, "inputTensor");
        this.e2eStartNs = e2eStartNs;
    }
}
