# import pandas as pd
# from dotmap import DotMap
# import matplotlib.pyplot as plt
# import yaml
# import numpy as np
# from PIL import Image
# import plotly.express as px
# from os.path import join
# from datetime import datetime as dt 


# def compare_plt(output,timeseries,fin_df, prop, scalelt):
#     for scale in scalelt:
#         args = yaml.load(open('./config/default.yaml'), Loader=yaml.FullLoader)
#         args = DotMap(args)
#         test_stat=pd.read_csv(args.IO.stats_file_path)
    
#         months_avgs=[]
#         sample_size0=[]
#         delta_mean=[]
#         i=1
#         for ts in timeseries:
#             smaple_avgs=[]
#             ts_gp=ts.groupby(pd.Grouper(freq='M'))
#             for n,g in ts_gp:
#                 df_re = g.resample(scale).sum()
#                 df_mean = df_re.mean()
#                 df_std = df_re.std()
#                 df_cv = df_std/df_mean
#                 df_skew = df_re.skew()
#                 if prop == 'Mean':
#                     smaple_avgs.append(df_mean)
#                 elif prop == 'CV':
#                     smaple_avgs.append(df_cv)
#                 elif prop == 'Skewness':
#                     smaple_avgs.append(df_skew)
#             sample_size0.append(len(smaple_avgs))
#             smaple_avgs=pd.DataFrame(smaple_avgs)
#             months_avgs.append(smaple_avgs)
#             i+=1
#         months_avgs=pd.concat(months_avgs,axis=1).dropna().values



#         # ===================================== Plot =================================
#         data = months_avgs
#         labels=['J','F','M','A','M','J','J','A','S','O','N','D']
#         plt.figure(figsize=(12,5))

#         ax = plt.axes()
#         ax.tick_params(axis='x', bottom=False)
#         xticks = ax.set_xticks(np.arange(len(labels) + 1) + 0.5, minor=True)
#         sample_sizes = [data.shape[0] for i in range(len(labels))]
#         sample_sizes = [f'{s:,}' for s in sample_size0]
#         x_pos = np.arange(len(labels)) + 1

#         minni = 9999
#         maxi = 0
#         for d in data:
#             for sig in d:
#                 if sig < minni:
#                     minni = sig
#                 if sig > maxi:
#                     maxi = sig

#         bp = ax.boxplot(
#             data, sym='', whis=[0, 100], widths=0.4, labels=labels,
#             patch_artist=False
#         )
        
#         stat_prop = str(prop) + '_' + str(scale)
#         test_mean=[round(s, 4) for s in list(test_stat.loc[:,stat_prop])]
#         sample_mean_findf=[round(s, 4) for s in list(fin_df.loc[:,stat_prop])]

#         delta_means=[abs(sample_mean_findf[i]-test_mean[i]) for i in range(len(sample_mean_findf))]
        
#         for t in test_mean:
#             if t < minni:
#                 minni = t
#             if t > maxi:
#                 maxi = t

#         for sam in sample_mean_findf:
#             if sam < minni:
#                 minni = sam
#             if sam > maxi:
#                 maxi = sam

#         if prop == 'Mean':
#             ax.set_ylim([minni - 0.05, maxi + 0.05])
#         else:
#             ax.set_ylim([minni - 1, maxi + 2])
#         i=0
#         # for xy in zip(range(1,13),test_mean):
            
#         #     ax.annotate(r'$\Delta\mu =' +str(round(delta_means[i],3))+'$',xy=xy,textcoords='offset points',rotation=60)
#         #     i+=1

#         y_pos = ax.get_ylim()[1]-0.05
#         # for i in range(len(labels)):
#         #     k = i % 2
#         #     ax.text(
#         #         x_pos[i], y_pos, r'$n =' + fr' {sample_sizes[i]}$',
#         #         horizontalalignment='center', size='small'
#         #     )
#         plt.scatter(range(1,len(test_stat.index[:])+1),test_stat.loc[:,stat_prop], label =  'obs', color = 'blue', marker = 'x')
#         plt.plot(range(1,len(fin_df.index[:])+1),fin_df.loc[:,stat_prop],  label =  'sample', color = 'green', marker = 'o')
#         plt.title(f'{output} {stat_prop}')
#         ax.legend()
        
#         ax.grid()
#         plt.savefig('./02 Output_data/Sampling_result/' + output + '/' + output + '_' + stat_prop + '.png',dpi=300)
