package benchmark.openvino;

import java.io.Serializable;

public final class DetectionResult implements Serializable {
    public final float maxScore;
    public final long e2eStartNs;

    public DetectionResult(float maxScore) {
        this(maxScore, 0L);
    }

    public DetectionResult(float maxScore, long e2eStartNs) {
        this.maxScore = maxScore;
        this.e2eStartNs = e2eStartNs;
    }

    @Override
    public String toString() {
        return "DetectionResult{" +
                "maxScore=" + maxScore +
                '}';
    }
}
