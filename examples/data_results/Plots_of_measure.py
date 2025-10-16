import scipy.stats as stats 
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dt = {
    # "names": ["environment","normalized_score_A","normalized_score_R","POIC","optimality_marginal","optimality_conditional","PIC","reward_marginal","reward_conditional","variance","temperatures","r_max","r_min","r_mean"],
    "1link100d": [0.9662310094152755,0.39347793319108504,0.0026278311610159477,0.6269332085391845,0.6243053773761686,4.005219391162119,10.142510531891979,6.13729114072986,20.106702650540164,62.8722201511843,-0.00023371097959601033,-158.7188153509334,-96.26655588822388],
    "1link165d": [0.9782714832084544,0.3363260288401702,0.004105344366139074,0.4212736690629056,0.4171683246947665,4.153200243014451,10.317925533155313,6.164725290140861,40.04700772234668,93.69267320999553,-0.0005418588771122082,-402.21198707465544,-266.937808951167],
    "2linkd": [0.9670827781428586,0.19588165691213727,0.0007224184258414201,0.4624184340596091,0.4616960156317677,4.205314483317399,10.40584737274708,6.200532889429682,8.90367908058127,114.12627726037114,-0.6328928030318617,-262.3085460347365,-211.0376102579028],
    "1link100s": [0.9196205913786961,0.24166608601046408,0.0019822805057017057,0.58004524351596,0.5780629630082583,0.08509590179758098,2.70138859355295,2.616292691755369,7.286923408281341,18.866748254552512,0.0,-44.912,-34.058292745098036],
    "1link165s": [0.9092093648858812,0.15168271326857008,0.0011966583362618133,0.36680230285306775,0.3656056445148059,0.07132927095146746,1.9875813705323393,1.9162520995808718,5.219586755263358,12.889605551788428,0.0,-47.582,-40.3646331372549],
    "2links": [0.06663194166199202,0.01820386877405017,0.0009463319619904056,0.006684352487955771,0.005738020523965387,0.045953113208955954,0.30571506351957023,0.2597619503106143,0.6638095119695409,0.004546304862749564,0.0,-49.916,-49.00733568627451],
}

# for key in dt.keys():
#     print(key)
#     print(np.asarray(dt[key]).shape)
#     # xx

# print(pd.DataFrame.from_dict(dt,))
# xxx
environments = ["normalized_score_A","normalized_score_R","POIC","optimality_marginal","optimality_conditional","PIC","reward_marginal","reward_conditional","variance","temperatures","r_max","r_min","r_mean"]
df = pd.DataFrame.from_dict(dt,orient='index')
df.columns = environments
print('df: \n', df)


v_df = df.transpose()
print('v_df: \n',v_df)

names=["1link100d","1link165d","2linkd","1link100s","1link165s","2links"]
colors=['k','r','g','b','m','c']


# corr,pval = pearsonr(df["POIC"],df["normalized_score_A"]) 
# print('corr: ', corr)
# print('pval: ',pval)
# xxx


#######
f_size = 13
f_size_legend = 10
val_size = 3
# figsize = [7*val_size,2*val_size]
# plt.figure(figsize=figsize)

# plt.subplot(2,7,1)
# for x, y, label,color in zip(df["POIC"],df["normalized_score_A"],names,colors):
#     plt.scatter(x,y,label=label,color=color,s=50)
# corr,pval = pearsonr(df["POIC"],df["normalized_score_A"]) 
# plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
# plt.ylabel("Normalized Score \n (Algorithm)",fontsize=f_size)
# # plt.xlabel("$\mathcal{\hat{I}}(\mathcal{O},\Theta)$",fontsize=f_size)
# plt.grid()
# # plt.legend(fontsize=f_size)

# plt.subplot(2,7,2)
# for x, y, label,color in zip(df["optimality_marginal"],df["normalized_score_A"],names,colors):
#     plt.scatter(x,y,label=label,color=color,s=50)
# corr,pval = pearsonr(df["optimality_marginal"],df["normalized_score_A"]) 
# plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
# # plt.xlabel("$\mathcal{\hat{H}(\mathcal{O})}$",fontsize=f_size)
# plt.grid()
# # plt.legend(fontsize=f_size)

# plt.subplot(2,7,3)
# for x, y, label,color in zip(df["optimality_conditional"],df["normalized_score_A"],names,colors):
#     plt.scatter(x,y,label=label,color=color,s=50)
# corr,pval = pearsonr(df["optimality_conditional"],df["normalized_score_A"]) 
# plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
# # plt.xlabel("$\mathcal{\hat{H}(\mathcal{O} \mid \Theta)}$",fontsize=f_size)
# plt.grid()
# # plt.legend(fontsize=f_size)

# plt.subplot(2,7,4)
# for x, y, label,color in zip(df["PIC"],df["normalized_score_A"],names,colors):
#     plt.scatter(x,y,label=label,color=color,s=50)
# corr,pval = pearsonr(df["PIC"],df["normalized_score_A"]) 
# plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
# # plt.xlabel("$\mathcal{\hat{I}}(R,\Theta)$",fontsize=f_size)
# plt.grid()

# plt.subplot(2,7,5)
# for x, y, label,color in zip(df["reward_marginal"],df["normalized_score_A"],names,colors):
#     plt.scatter(x,y,label=label,color=color,s=50)
# corr,pval = pearsonr(df["reward_marginal"],df["normalized_score_A"]) 
# plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
# # plt.xlabel("$\mathcal{\hat{H}(R)}$",fontsize=f_size)
# plt.grid()
# # plt.legend(fontsize=f_size)


# plt.subplot(2,7,6)
# for x, y, label,color in zip(df["reward_conditional"],df["normalized_score_A"],names,colors):
#     plt.scatter(x,y,label=label,color=color,s=50)
# corr,pval = pearsonr(df["reward_conditional"],df["normalized_score_A"]) 
# plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
# # plt.xlabel("$\mathcal{\hat{H}(R \mid \Theta)}$",fontsize=f_size)
# plt.grid()
# # plt.legend(fontsize=f_size)

# plt.subplot(2,7,7)
# for x, y, label,color in zip(df["variance"],df["normalized_score_A"],names,colors):
#     plt.scatter(x,y,label=label,color=color,s=50)
# corr,pval = pearsonr(df["variance"],df["normalized_score_A"]) 
# plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
# # plt.xlabel("Variance(Reward)",fontsize=f_size)
# plt.grid()
# plt.legend(loc="center left",bbox_to_anchor=(1,0.5),fontsize=f_size_legend)

# ##################################################################


# plt.subplot(2,7,8)
# for x, y, label,color in zip(df["POIC"],df["normalized_score_R"],names,colors):
#     plt.scatter(x,y,label=label,color=color,s=50)
# corr,pval = pearsonr(df["POIC"],df["normalized_score_R"]) 
# plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
# plt.ylabel("Normalized Score \n (Random Sampling)",fontsize=f_size)
# plt.xlabel("$\mathcal{\hat{I}}(\mathcal{O},\Theta)$",fontsize=f_size)
# plt.grid()
# # plt.legend(fontsize=f_size)

# plt.subplot(2,7,9)
# for x, y, label,color in zip(df["optimality_marginal"],df["normalized_score_R"],names,colors):
#     plt.scatter(x,y,label=label,color=color,s=50)
# corr,pval = pearsonr(df["optimality_marginal"],df["normalized_score_R"]) 
# plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
# plt.xlabel("$\mathcal{\hat{H}(\mathcal{O})}$",fontsize=f_size)
# plt.grid()
# # plt.legend(fontsize=f_size)

# plt.subplot(2,7,10)
# for x, y, label,color in zip(df["optimality_conditional"],df["normalized_score_R"],names,colors):
#     plt.scatter(x,y,label=label,color=color,s=50)
# corr,pval = pearsonr(df["optimality_conditional"],df["normalized_score_R"]) 
# plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
# plt.xlabel("$\mathcal{\hat{H}(\mathcal{O} \mid \Theta)}$",fontsize=f_size)
# plt.grid()
# # plt.legend(fontsize=f_size)

# plt.subplot(2,7,11)
# for x, y, label,color in zip(df["PIC"],df["normalized_score_R"],names,colors):
#     plt.scatter(x,y,label=label,color=color,s=50)
# corr,pval = pearsonr(df["PIC"],df["normalized_score_R"]) 
# plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
# plt.xlabel("$\mathcal{\hat{I}}(R,\Theta)$",fontsize=f_size)
# plt.grid()

# plt.subplot(2,7,12)
# for x, y, label,color in zip(df["reward_marginal"],df["normalized_score_R"],names,colors):
#     plt.scatter(x,y,label=label,color=color,s=50)
# corr,pval = pearsonr(df["reward_marginal"],df["normalized_score_R"]) 
# plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
# plt.xlabel("$\mathcal{\hat{H}(R)}$",fontsize=f_size)
# plt.grid()
# # plt.legend(fontsize=f_size)


# plt.subplot(2,7,13)
# for x, y, label,color in zip(df["reward_conditional"],df["normalized_score_R"],names,colors):
#     plt.scatter(x,y,label=label,color=color,s=50)
# corr,pval = pearsonr(df["reward_conditional"],df["normalized_score_R"]) 
# plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
# plt.xlabel("$\mathcal{\hat{H}(R \mid \Theta)}$",fontsize=f_size)
# plt.grid()
# # plt.legend(fontsize=f_size)

# plt.subplot(2,7,14)
# for x, y, label,color in zip(df["variance"],df["normalized_score_R"],names,colors):
#     plt.scatter(x,y,label=label,color=color,s=50)
# corr,pval = pearsonr(df["variance"],df["normalized_score_R"]) 
# plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
# plt.xlabel("Variance(Reward)",fontsize=f_size)
# plt.grid()
# plt.legend(loc="center left",bbox_to_anchor=(1,0.5),fontsize=f_size_legend)

# #############################################################################################
figsize = [7*val_size,val_size]
plt.figure(figsize=figsize)

plt.subplot(1,7,1)
for x, y, label,color in zip(df["POIC"],df["normalized_score_R"],names,colors):
    plt.scatter(x,y,label=label,color=color,s=50)
corr,pval = pearsonr(df["POIC"],df["normalized_score_R"]) 
plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
plt.ylabel("Normalized Score \n (Random Sampling)",fontsize=f_size)
plt.xlabel("$\mathcal{\hat{I}}(\mathcal{O},\Theta)$",fontsize=f_size)
plt.grid()
# plt.legend(fontsize=f_size)

plt.subplot(1,7,2)
for x, y, label,color in zip(df["optimality_marginal"],df["normalized_score_R"],names,colors):
    plt.scatter(x,y,label=label,color=color,s=50)
corr,pval = pearsonr(df["optimality_marginal"],df["normalized_score_R"]) 
plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
plt.xlabel("$\mathcal{\hat{H}(\mathcal{O})}$",fontsize=f_size)
plt.grid()
# plt.legend(fontsize=f_size)

plt.subplot(1,7,3)
for x, y, label,color in zip(df["optimality_conditional"],df["normalized_score_R"],names,colors):
    plt.scatter(x,y,label=label,color=color,s=50)
corr,pval = pearsonr(df["optimality_conditional"],df["normalized_score_R"]) 
plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
plt.xlabel("$\mathcal{\hat{H}(\mathcal{O} \mid \Theta)}$",fontsize=f_size)
plt.grid()
# plt.legend(fontsize=f_size)

plt.subplot(1,7,4)
for x, y, label,color in zip(df["PIC"],df["normalized_score_R"],names,colors):
    plt.scatter(x,y,label=label,color=color,s=50)
corr,pval = pearsonr(df["PIC"],df["normalized_score_R"]) 
plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
plt.xlabel("$\mathcal{\hat{I}}(R,\Theta)$",fontsize=f_size)
plt.grid()

plt.subplot(1,7,5)
for x, y, label,color in zip(df["reward_marginal"],df["normalized_score_R"],names,colors):
    plt.scatter(x,y,label=label,color=color,s=50)
corr,pval = pearsonr(df["reward_marginal"],df["normalized_score_R"]) 
plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
plt.xlabel("$\mathcal{\hat{H}(R)}$",fontsize=f_size)
plt.grid()
# plt.legend(fontsize=f_size)


plt.subplot(1,7,6)
for x, y, label,color in zip(df["reward_conditional"],df["normalized_score_R"],names,colors):
    plt.scatter(x,y,label=label,color=color,s=50)
corr,pval = pearsonr(df["reward_conditional"],df["normalized_score_R"]) 
plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
plt.xlabel("$\mathcal{\hat{H}(R \mid \Theta)}$",fontsize=f_size)
plt.grid()
# plt.legend(fontsize=f_size)

plt.subplot(1,7,7)
for x, y, label,color in zip(df["variance"],df["normalized_score_R"],names,colors):
    plt.scatter(x,y,label=label,color=color,s=50)
corr,pval = pearsonr(df["variance"],df["normalized_score_R"]) 
plt.title(f"R = {corr:.3f}, p = {pval:.3f}",fontsize=f_size)
plt.xlabel("Variance(Reward)",fontsize=f_size)
plt.grid()
plt.legend(loc="center left",bbox_to_anchor=(1,0.5),fontsize=f_size_legend)



plt.tight_layout()
# save the figure
plt.savefig(
    'scatter_plots.png', 
    dpi=300, 
    bbox_inches='tight',
    format='png')


plt.show()