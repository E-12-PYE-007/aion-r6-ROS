std_msgs/Header header
# Frame sys2 ran its forward pass on -- older than curr_img_seq_num below. sys1 keys its
# image buffer on this to find the same observation, so it must equal the publishing sys2's
# HiddenState.img_seq_num exactly: frames are matched by seq num, never by header stamp.
uint32 old_img_seq_num
# Newest frame this chunk was conditioned on. Equals old_img_seq_num for publishers with a
# single observation per chunk (sys2); sys1 conditions on both frames, since it runs faster
# than sys2, and the gap between the two spans one sys2 inference.
uint32 curr_img_seq_num
geometry_msgs/Pose2D[8] relative_poses
