import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_with_customizations(df1, df2, dataset,tuner,model1_label='OPT125M', model2_label='OPT350M'):
    fig, ax = plt.subplots(figsize=(10, 5))


    ax.scatter(df1['in_domain_accuracy'], df1['out_of_domain_accuracy'], alpha=0.5, label=model1_label)
    if df2 is not None:
        ax.scatter(df2['in_domain_accuracy'], df2['out_of_domain_accuracy'], alpha=0.5, label=model2_label, marker='x')

    # Plot a dashed equality line for reference
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # Find the lower limit across both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # Find the upper limit across both axes
    ]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)  # Black dashed line (k--)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Labeling
    ax.grid(True)

    ax.set_xlabel('In-domain Accuracy')
    ax.set_ylabel('Out-of-domain Accuracy')
    ax.set_title(f"{dataset}: {tuner}")
    #fig.suptitle(tuner, fontsize=16, ha='left', x=0.125)

    # Set legend
    ax.legend()
    folder_path = "C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/plots"  # Replace with the path to your desired folder
    plt.savefig(f"{folder_path}/{dataset}_{tuner}_all.png", bbox_inches='tight')

    #plt.savefig(f"{dataset}_{tuner}_all.png", bbox_inches='tight')
    plt.show()


# ********************************************************************************
# COLA VANILLA

df_125M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/vanilla_cola_baseline_125M.csv')
df_350M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/vanilla_cola_baseline_350M.csv')


dataset = "COLA dataset"
tuner = "Vanilla tuning"
avg_125M = df_125M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
avg_350M = df_350M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
plot_accuracy_with_customizations(avg_125M, avg_350M,dataset, tuner)


# ********************************************************************************
# COLA PBFT
df_125M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/pbft_cola_baseline.ipynb.csv')
df_350M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/pbft_cola_baseline_350m.csv')

dataset = "COLA dataset"
tuner = "PBFT tuning"
avg_125M = df_125M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
avg_350M = df_350M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
plot_accuracy_with_customizations(avg_125M, avg_350M,dataset, tuner)


# ********************************************************************************
# COLA PEFT with LORA
df_125M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/peft_LoRA_on_cola.csv')

dataset = "COLA dataset"
tuner = "PEFT with LORA tuning"
avg_125M = df_125M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
avg_350M = df_350M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
plot_accuracy_with_customizations(avg_125M, None,dataset, tuner)

# ********************************************************************************
# COLA Adaptive Tuning
df_125M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/vanilla_cola_adaptive_125M_2Layer_LearningRate.csv')

dataset = "COLA dataset"
tuner = "Adaptive_Tuning"
avg_125M = df_125M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
avg_350M = df_350M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
plot_accuracy_with_customizations(avg_125M, None,dataset, tuner)

# ********************************************************************************
# context_distillation_mnli.csv

# # COLA Adaptive Tuning
# df_125M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/context_distillation_mnli.csv')
#
# dataset = "MNLI dataset"
# tuner = "Context Distillation"
# avg_125M = df_125M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
# avg_350M = df_350M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
# plot_accuracy_with_customizations(avg_125M, None,dataset, tuner)

# ********************************************************************************
# few_shot_context_distillation_mnli_baseline_results.csv

# COLA Adaptive Tuning
df_125M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/few_shot_context_distillation_mnli_baseline_results.csv')

dataset = "MNLI dataset"
tuner = "Few Shot Context Distillation"
avg_125M = df_125M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
avg_350M = df_350M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
plot_accuracy_with_customizations(avg_125M, None,dataset, tuner)

# ********************************************************************************
# MNLI  PBFT Baseline

df_125M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/pbft_mnli_baseline_hansOOD.csv')
df_350M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/pbft_mnli_baseline_hansOOD_350m.csv')


dataset = "MNLI dataset"
tuner = "PBFT tuning"
avg_125M = df_125M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
avg_350M = df_350M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
plot_accuracy_with_customizations(avg_125M, avg_350M,dataset, tuner)

# ********************************************************************************
# MNLI  PBFT Baseline

df_125M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/vanilla_mnli_baseline_results_hans.csv')
df_350M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/vanilla_mnli_baseline_results_hans_350M.csv')


dataset = "MNLI dataset"
tuner = "Vanilla tuning"
avg_125M = df_125M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
avg_350M = df_350M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
plot_accuracy_with_customizations(avg_125M, avg_350M,dataset, tuner)









