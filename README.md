Soft Mean Expected Calibration Error (SMECE): A Calibration Metric for Probabilistic Labels
Michael Leznik
https://arxiv.org/abs/2603.14092

The Expected Calibration Error (ece), the dominant calibration metric in machine learning, compares predicted probabilities against empirical 
frequencies of binary outcomes. This is appropriate when labels are binary events. However, many modern settings produce labels that are 
themselves probabilities rather than binary outcomes: a radiologist's stated confidence, a teacher model's soft output in knowledge distillation, 
a class posterior derived from a generative model, or an annotator agreement fraction. In these settings, ece commits a category error - it discards 
the probabilistic information in the label by forcing it into a binary comparison. The result is not a noisy approximation that more data will correct. 
It is a structural misalignment that persists and converges to the wrong answer with increasing precision as the sample size grows. We introduce the Soft 
Mean Expected Calibration Error (smece), a calibration metric for settings where labels are of a probabilistic nature. The modification to the ece 
formula is one line: replace the empirical hard-label fraction in each prediction bin with the mean probability label of the samples in that bin. 
smece reduces exactly to ece when labels are binary, making it a strict generalisation.

