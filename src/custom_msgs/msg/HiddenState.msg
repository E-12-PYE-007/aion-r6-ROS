std_msgs/Header header
# The frame sys2 ran on (the paper's timestamp of I_{t-k}, returned so the edge
# controller can find the observation used for inference).
int32 img_seq_num
# Newest /cam frame to arrive while sys2's forward pass was still running; the gap from
# img_seq_num is how far the world moved during inference.
int32 end_img_seq_num
# Duration of that forward pass, on the node clock. Different epoch from the /cam header
# stamps, which are sim time -- compare frames by seq_num, not by stamp.
float32 inference_ms

std_msgs/Float32MultiArray hidden_states
