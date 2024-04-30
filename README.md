# CS7643 Project (Georgia Institute of Technology) -  Efficient Fine Tuning of Large Language Models

## Abstract from our paper
In the domain of large language models (LLMs), McCoy et al. [2019] showed that few-shot full-model fine-tuning – namely Vanilla Fine Tuning (FT) and Pattern-Based Fine Tuning (PBFT) –, and In-Context Learning (ICL) generalize similarly on Out-Of-Domain (OOD) datasets, but vary in terms of task adaptation. However, they both pose challenges, especially in term of memory requirements. In this paper, we further try to push the understanding of different fine-tuning strategies for LLM and aim to bring a myriad of these on the same pedestal for an elaborate comparison with full-model fine-tuning on two diverse datasets. To that end, we conducted a series of experiments, beginning with state-of-the-art methods like vanilla fine-tuning and Pattern-Based Fine-Tuning (PBFT) on pre-trained models across two datasets, COLA and MNLI. We then investigate adaptive fine-tuning and the efficiency of LoRA adapters in a few-shot setting. Finally, we also compare an alternative approach that has gained recent popularity – context distillation – with the vanilla FT and PBFT in and without few-shot setup.

Our findings suggest that these alternative strategies that we explored can exhibit out-of-domain generalization comparable to that of vanilla FT and PBFT. PBFT underperforms Vanilla FT on out-of-domain (OOD) data, emphasizing the need for effective prompts. Further, our adaptivefine tuning and LoRA experiments perform comparable or slightly worse than the standard fine-tunings as anticipated, since standard fine-tunings involve tuning the entire model. Finally, our context distillation experiments out-perform the standard fine-tuning methods. These findings underscore that eventually the choice of an appropriate fine-tuning method depends on the available resources (memory, compute, data) and task adaptability.

## Codebase organization
The codebase is segregated into different folders in the following manner:

1. *Notebooks* - contains all the different notebooks for different experiments that we run with 

2. *Results* - contains all the output `.csv` file from all the experiments

3. *datafiles* - contains any neccessary data used for model inference

4. *requirements.txt* - has the list of all the necessary libraries used by this codebase

## Authors

*all authors are equal contributors*

1. Prasanth Gumpena (pgumpena3@gatech.edu)

2. Madhusudhana Yattapu (madhu.yattapu@gatech.edu)

3. Vishal Harshadray Brahmbhatt (vbrahmbhatt3@gatech.edu)

4. Krishna Prasad Varadarajan Srinivasan (kvaradar3@gatech.edu)
