# Constitutional
This repository is an independent reimplementation inspired by "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022).





The project explores whether a language model can be trained to exhibit greater epistemic humility using a simplified Constitutional AI–style self-critique loop. The model generates ambiguous or underspecified user questions, produces initial responses, critiques those responses along an epistemic calibration axis (overconfidence vs appropriate uncertainty), and then revises its answers accordingly. The revised responses are used as synthetic supervision for fine-tuning. The goal is not benchmark performance, but to test whether self-generated feedback can reliably shift a specific behavioral property without human labeling.



![Epistemic Humility Rating per Epoch](data/plots/mean_epistemic_humility_rating_per_epoch_20260110_050220.svg)

Figure 1. A graph of Epistemic Humility Rating (as marked by a frozen judge model) over each training epoch of a qwen3-4B model in the SL-CAI loop. There is a modest increase at the first epooch, followed by a plateau and then a fall off. When fitted with the trained LoRA (epochs 1+), the PEFT model exhibits consistently higher Epistemic Humility Ratings than the base model (epoch 0).
