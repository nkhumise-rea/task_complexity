import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('agent_data.csv')

def convert_data(df):
    window_size = 2
    mean = df.rolling(window_size).mean()
    var = df.rolling(window_size).std()
    #print('mean: ', mean)
    #print('var: ', var)

    lower_bound = mean - var
    upper_bound = mean + var
    #print('lower_bound: ', lower_bound['agent'].values[1:])
    #print('upper_bound: ', upper_bound)
    return lower_bound['agent'].values[1:], \
            upper_bound['agent'].values[1:], \
            mean['agent'].values[1:]      

def plotting(data):
    agent  = data
    print('agent: ',agent)

    ## Plots
    plt.figure(figsize=[17,15])
    plt.subplot(3,2,1)

    age_lower_bound, age_upper_bound, agent_mean = convert_data(agent)
    episodes = np.arange(0,age_lower_bound.shape[0],1)
    print(episodes)

    plt.plot(agent_mean,color="tomato",label='agent'  )
    plt.fill_between(episodes,
                    age_lower_bound,
                    age_upper_bound,
                    facecolor="tomato", 
                    alpha=0.15,
                    #label='std'
                    )
        
    plt.legend()
    plt.ylabel('running_score')
    plt.grid()
    plt.xlabel('episodes')
    plt.show()


##Run

plotting(df)

