# Few-Shot Learning for Medical Image Segmentation

A PyTorch project exploring binary segmentation of abdominal MRI slices with a **U-Net** and **Reptile meta-learning**. The model learns an initialization across training tasks, then adapts to a held-out task using a small set of labelled support images.

This work was originally completed and submitted as a group project for **EMS741: Deep Learning for Data and Image Analysis** at **Queen Mary University of London**. This repository preserves the submitted notebook for reproduction and GitHub project management.

## Notebook

The main implementation is [EMS741_Original_University_Notebook_GitHub_Data.ipynb](EMS741_Original_University_Notebook_GitHub_Data.ipynb).

Its model, loss functions, data loading, meta-training, evaluation and plotting code are preserved from the university submission. The setup cell now installs the required libraries directly and downloads the original dataset from this repository's GitHub Release, replacing the unavailable group repository and original download route.

## Run in Google Colab

1. [Open the notebook in Google Colab](https://colab.research.google.com/github/TheSkyEr4998/Few-shot-Learning-for-Medical-Image-Segmentation/blob/main/EMS741_Original_University_Notebook_GitHub_Data.ipynb). Alternatively, upload the `.ipynb` file directly to Colab. The repository link becomes available after the notebook is pushed to `main`.
2. Select **Runtime → Change runtime type → Python 3 → T4 GPU**, then save.
3. Clear previous outputs when recording a fresh run, and run the cells from top to bottom.
4. The setup cell installs dependencies, downloads the ZIP and extracts the supplied train, validation and test folders under `/content/ems741_cw_data`.
5. Allow training to finish before running the testing and prediction cells.
6. Download the executed notebook, model checkpoint and generated plots before ending the Colab session.

The dependencies used by the notebook are **PyTorch, torchvision, NumPy, Pillow and Matplotlib**. The full reproduction described below took **147 minutes for training on a Tesla T4**; setup, testing and plotting add to that time. Runtime varies between sessions.

## Dataset

The dataset was supplied for the university coursework. It contains grayscale images and corresponding binary segmentation masks organized into eight numbered tasks.

Download [ems741_cw_data.zip from the data-v1 release](https://github.com/TheSkyEr4998/Few-shot-Learning-for-Medical-Image-Segmentation/releases/download/data-v1/ems741_cw_data.zip), or let the notebook download it automatically.

The original task split is preserved:

| Split | Tasks | Image–mask pairs |
| --- | --- | ---: |
| Training | `task_2`, `task_3`, `task_5`, `task_7` | 3,576 |
| Validation | `task_4`, `task_6` | 2,389 |
| Test | `task_1`, `task_8` | 1,423 |
| **Total** | **8 tasks** | **7,388** |

Each task contains an `images/` folder and a `masks/` folder, with corresponding files paired by filename. Some slices have empty target masks.

Keep the dataset ZIP and extracted data outside Git tracking. The ZIP is distributed as a Release asset, and `/ems741_cw_data.zip` is excluded through `.gitignore`.

## Model and training

The U-Net uses skip connections, four downsampling stages and four upsampling stages. It produces one foreground logit per pixel. Images are resized using bilinear interpolation; masks use nearest-neighbour interpolation and are binarized at a threshold greater than 0.5.

For each meta-iteration, the notebook samples training tasks, copies the current model, and adapts each copy on its labelled support set. The Reptile update moves the shared initialization toward the adapted model weights. Validation queries are used to select the checkpoint with the highest mean validation Dice.

| Setting | Value |
| --- | --- |
| Input | Grayscale, 192 × 192 pixels |
| U-Net base channels | 32 |
| Model parameters | 7,759,521 |
| Loss | 0.5 × binary cross-entropy + 0.5 × soft Dice loss |
| Meta-iterations | 800 |
| Meta-batch size | 4 tasks, sampled with replacement |
| Training support set | 3 labelled image–mask pairs per task |
| Inner adaptation | 20 Adam steps at learning rate 0.003 |
| Outer interpolation step | Linearly decreases from 0.15 to 0.01 |
| Validation | Iteration 1 and every 50 iterations; two episodes per validation task |
| Checkpoint selection | Highest validation Dice |
| Prediction threshold | Sigmoid probability greater than 0.5 |
| Initial seed | 42 |

Final testing adapts the selected checkpoint using **1, 3, 5 and 10 support examples**. Each setting uses five episodes per test task, ten query slices per episode, and the same 20-step Adam adaptation procedure.

## Latest observed reproduction results

The following values were recorded from the completed Colab rerun of the original notebook using the GitHub-hosted dataset. They describe one observed run, rather than guaranteed scores for every execution.

- **GPU:** Tesla T4.
- **Training duration:** 147.0 minutes.
- **Best validation Dice:** 0.4443 at meta-iteration 250.
- **Training completed:** all 800 meta-iterations.
- **Testing checkpoint:** the best-validation model, rather than the final-iteration model.

### Main test evaluation

| Support examples | Mean Dice | Mean IoU |
| --- | ---: | ---: |
| 1-shot | 0.1410 | 0.0992 |
| 3-shot | 0.2288 | 0.1689 |
| **5-shot** | **0.2808** | **0.2067** |
| 10-shot | 0.2569 | 0.1869 |

**5-shot achieved the highest mean Dice in this evaluation.** Increasing the support count did not consistently improve performance. Some predictions miss foreground regions or produce substantial false positives, and the separate per-task evaluation shows weaker performance on `task_8` than on `task_1`.

The supplied notebook retains historical output cells from the university submission. Those older outputs can differ from the latest results above; rerun the notebook and save its executed version when reporting a new experiment.

## Interpreting the evaluation

- Dice and IoU measure overlap between the predicted binary mask and the ground truth. Scores are averaged within episodes, then across task-repeat episodes.
- When both prediction and target are empty, the original metrics exclude that query image. An episode with every query excluded returns zero. This convention is retained for reproduction.
- The notebook's `PASS` label means only **Dice > 0.2**, the threshold defined in its code.
- The main test table and later per-task table sample different episodes. Their random seeds are respectively `42 + 1000 × task_index + repeat_index` and `42 + 1000 × repeat_index`, so their averages need not match. The per-task table is a separate evaluation.
- Changing the support count also changes the sampled queries in the main numerical evaluation. The separate qualitative shot-comparison plot uses fixed queries.
- The supplied split separates task IDs. Patient-level independence is not established by this split or by the notebook.
- The supplied notebook implements the U-Net/Reptile experiment. It does not include the supervised baseline implementation discussed in the group report, so these results do not establish an improvement over that baseline.

## Generated outputs

The notebook saves these files in the Colab `/content` directory:

| File | Contents |
| --- | --- |
| `best_reptile_unet.pt` | Model weights selected by validation Dice |
| `sample_data.png` | Example images and masks from each split |
| `episode_example.png` | A few-shot support/query episode |
| `training_curves.png` | Validation loss, Dice and IoU over training |
| `kshot_bar_chart.png` | Main test Dice and IoU by support count |
| `predictions_1shot.png`, `predictions_3shot.png`, `predictions_5shot.png` | Predictions across test tasks |
| `shot_comparison.png` | Predictions on fixed queries with different support counts |
| `per_task_dice.png` | Dice scores from the separate per-task evaluation |

The original checkpoint stores model weights. The notebook does not save the full state needed to resume an interrupted training run.

## Credits

The original university report credits:

- Alexis Adebayo
- Niranjan Ajay Ghone
- Omkar Vilas Kasabe
- Shree Narendra Korde
- Abdul Mirza

The original experiment is group work. This GitHub copy is maintained by [Niranjan Ajay Ghone](https://github.com/TheSkyEr4998), with updated dataset access for rerunning the submission.

## Method references

1. Olaf Ronneberger, Philipp Fischer and Thomas Brox. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), 2015.
2. Alex Nichol, Joshua Achiam and John Schulman. [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999), 2018. Introduces the Reptile algorithm.
