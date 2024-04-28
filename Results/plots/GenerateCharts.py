import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ************************************************************************************
#  Version 1  : with no legend for "n"
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
    folder_path = "C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/plots"
    plt.savefig(f"{folder_path}/{dataset}_{tuner}_all.png", bbox_inches='tight')

    #plt.savefig(f"{dataset}_{tuner}_all.png", bbox_inches='tight')
    plt.show()


# ************************************************************************************
# Includes legend for n
def plot_accuracy_with_customizations2(df1, df2, dataset, tuner, model1_label='OPT125M', model2_label='OPT350M'):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot individual points with labels from df1
    for _, row in df1.iterrows():
        ax.scatter(row['in_domain_accuracy'], row['out_of_domain_accuracy'], alpha=0.5,
                   label=f'{model1_label} n={int(row["n"])}')

    # If a second dataframe is provided, plot its points as well
    if df2 is not None:
        for _, row in df2.iterrows():
            ax.scatter(row['in_domain_accuracy'], row['out_of_domain_accuracy'], alpha=0.5,
                       label=f'{model2_label} n={int(row["n"])}', marker='x')  # Convert n to integer

    # Plot a dashed equality line for reference
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Labeling
    ax.grid(True)
    ax.set_xlabel('In-domain Accuracy')
    ax.set_ylabel('Out-of-domain Accuracy')
    ax.set_title(f"{dataset}: {tuner}")

    # Set legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
    plt.tight_layout()

    folder_path = "C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/plots"
    plt.savefig(f"{folder_path}/{dataset}_{tuner}_all.png", bbox_inches='tight')

    plt.show()

# ********************************************************************************
# COLA VANILLA

df_125M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/vanilla_cola_baseline_125M.csv')
df_350M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/vanilla_cola_baseline_350M.csv')


dataset = "COLA dataset"
tuner = "Vanilla tuning"
avg_125M = df_125M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
avg_350M = df_350M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
plot_accuracy_with_customizations2(avg_125M, avg_350M,dataset, tuner)


# ********************************************************************************
# COLA PBFT
df_125M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/pbft_cola_baseline.ipynb.csv')
df_350M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/pbft_cola_baseline_350m.csv')

dataset = "COLA dataset"
tuner = "PBFT tuning"
avg_125M = df_125M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
avg_350M = df_350M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
plot_accuracy_with_customizations2(avg_125M, avg_350M,dataset, tuner)


# ********************************************************************************
# COLA PEFT with LORA
df_125M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/peft_LoRA_on_cola.csv')

dataset = "COLA dataset"
tuner = "PEFT with LORA tuning"
avg_125M = df_125M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
avg_350M = df_350M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
plot_accuracy_with_customizations2(avg_125M, None,dataset, tuner)

# ********************************************************************************
# COLA Adaptive Tuning
df_125M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/vanilla_cola_adaptive_125M_2Layer_LearningRate.csv')

dataset = "COLA dataset"
tuner = "Adaptive_Tuning"
avg_125M = df_125M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
avg_350M = df_350M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
plot_accuracy_with_customizations2(avg_125M, None,dataset, tuner)



# ********************************************************************************
# few_shot_context_distillation_mnli_baseline_results.csv

# COLA Adaptive Tuning
df_125M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/few_shot_context_distillation_mnli_baseline_results.csv')

dataset = "MNLI dataset"
tuner = "Few Shot Context Distillation"
avg_125M = df_125M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
avg_350M = df_350M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
plot_accuracy_with_customizations2(avg_125M, None,dataset, tuner)

# ********************************************************************************
# MNLI  PBFT Baseline

df_125M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/pbft_mnli_baseline_hansOOD.csv')
df_350M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/pbft_mnli_baseline_hansOOD_350m.csv')


dataset = "MNLI dataset"
tuner = "PBFT tuning"
avg_125M = df_125M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
avg_350M = df_350M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
plot_accuracy_with_customizations2(avg_125M, avg_350M,dataset, tuner)

# ********************************************************************************
# MNLI  PBFT Baseline

df_125M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/vanilla_mnli_baseline_results_hans.csv')
df_350M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/vanilla_mnli_baseline_results_hans_350M.csv')


dataset = "MNLI dataset"
tuner = "Vanilla tuning"
avg_125M = df_125M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
avg_350M = df_350M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
plot_accuracy_with_customizations2(avg_125M, avg_350M,dataset, tuner)

# ********************************************************************************
# context_distillation_mnli.csv
# seperate plot to hand no "N"

def plot_accuracy_with_context(df1, dataset,tuner, model1_label='OPT125M'):
    fig, ax = plt.subplots(figsize=(10, 5))

    for index, row in df1.iterrows():
        ax.scatter(row['in_domain_accuracy'], row['out_of_domain_accuracy'], alpha=0.5,
                   label=f"{model1_label} Iter {index + 1}")

    # Plot a dashed equality line for reference
    lims = [
        np.min([np.min(df1['in_domain_accuracy']), np.min(df1['out_of_domain_accuracy'])]),
        np.max([np.max(df1['in_domain_accuracy']), np.max(df1['out_of_domain_accuracy'])])
    ]
    # Add a small margin to the limits to ensure visibility of points near the axes
    margin = (lims[1] - lims[0]) * 0.05  # 5% of the range as margin
    lims_with_margin = [lims[0] - margin, lims[1] + margin]

    ax.plot(lims_with_margin, lims_with_margin, 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims_with_margin)
    ax.set_ylim(lims_with_margin)

    # Labeling
    ax.grid(True)
    ax.set_xlabel('In-domain Accuracy')
    ax.set_ylabel('Out-of-domain Accuracy')
    ax.set_title(f"{dataset}: {tuner}")

    # Set legend
    # Set legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
    plt.tight_layout()

    folder_path = "C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/plots"
    plt.savefig(f"{folder_path}/{dataset}_{tuner}_all.png", bbox_inches='tight')

    plt.show()





# COLA Adaptive Tuning
df_125M = pd.read_csv('C:/Users/Siddhu/PycharmProjects/CS7643/llm_finetuning/Results/context_distillation_mnli.csv')

dataset = "MNLI dataset"
tuner = "Context Distillation"
#avg_125M = df_125M.groupby('n').agg({'in_domain_accuracy':'mean', 'out_of_domain_accuracy':'mean'}).reset_index()
#plot_accuracy_with_context(df_125M,dataset,tuner)
dataset = "MNLI dataset"
tuner = "Context Distillation"
plot_accuracy_with_context(df_125M,dataset,tuner)







