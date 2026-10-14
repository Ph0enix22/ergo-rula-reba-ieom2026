# Project: Automated RULA/REBA Posture Scoring (research paper software)

## Goal
Build a Python system that takes images/video of a worker and outputs RULA and REBA
scores, comparing three pipelines on identical data. This supports a conference paper
(IEOM 2026), so rigor and reproducibility matter more than UI polish.

## Pipelines (all share one rule engine + one eval harness)
- Arm A: MediaPipe Pose (or YOLOv8-Pose) -> joint angles -> rule engine
- Arm B: ResNet-50-based keypoint detector (torchvision keypointrcnn_resnet50_fpn, or
  ResNet-50 backbone + heatmap head fine-tuned) -> joint angles -> rule engine
- Arm C: ResNet-50 (ImageNet-pretrained) fine-tuned end-to-end to predict the RULA grand
  score and REBA score (classification of risk levels + score regression variant)

## Components
1. rula_reba/angles.py: compute neck, trunk, upper arm, lower arm, wrist, leg angles from
   2D keypoints (document 2D-projection limits; support side-view and front-view).
2. rula_reba/rules.py: faithful RULA (Tables A, B, C) and REBA (Tables A, B, C) lookups,
   with unit tests against hand-worked examples from the official worksheets.
   Inputs that cannot come from vision (wrist twist, load/force, muscle-use/repetition)
   must be explicit config parameters, not silently guessed.
3. models/: the three arms behind a common interface predict(frame) -> keypoints/scores.
4. data/: loaders for (a) public pose datasets for keypoint fine-tuning/eval (COCO
   keypoints, MPII), (b) my own labelled set: images + CSV with human RULA/REBA scores
   and optionally joint-angle labels. NEVER fabricate or synthesize labels silently;
   if pseudo-labels are generated from the rule engine, flag them clearly as pseudo-labels
   and keep them out of the test set.
5. eval/: identical train/val/test splits (fixed seed, split by subject/video not by frame).
   Metrics: joint-angle MAE (deg), PCK/MPJPE for keypoints, score accuracy, +-1 score
   accuracy, weighted Cohen's kappa vs human labels, confusion matrix, per-risk-level F1,
   inference latency + FPS (report hardware), robustness tests (occlusion masks,
   brightness/blur changes, camera-angle variation). Output results to results/*.csv and
   auto-generated plots.
6. app/: a Streamlit demo (upload image/video -> skeleton overlay + angles + RULA/REBA
   score + risk level), with a privacy toggle that blurs faces and keeps all processing
   local.
7. configs/*.yaml for every experiment; scripts/train_*.py, scripts/evaluate.py;
   README with exact reproduction commands.

## Constraints
- Python 3.10+, PyTorch, OpenCV. Set seeds. Log every run.
- Small, readable modules; type hints; pytest for angle math and rule tables.
- Be honest in results: report failures and limits (self-occlusion, monocular depth
  error, lighting). Don't tune on the test set.

## Process
1. First produce a plan with milestones and flag any assumption or missing data.
2. Build in this order: rules.py + tests -> angles.py -> Arm A end-to-end on sample
   images -> eval harness -> Arm B -> Arm C -> Streamlit demo.
3. After each milestone, run tests and show me the output before moving on.
4. Ask me when you need data, labels, or a design decision. Don't invent them.