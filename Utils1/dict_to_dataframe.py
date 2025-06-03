import pandas as pd
import numpy as np

# The dictionary with average polarity scores
data = {
    'The Shawshank Redemption': {
        'question1': {'average_negative': np.float32(0.78929555), 'average_neutral': np.float32(0.15793686), 'average_positive': np.float32(0.052767534)},
        'question2': {'average_negative': np.float32(0.080560304), 'average_neutral': np.float32(0.15767983), 'average_positive': np.float32(0.7617598)},
        'question3': {'average_negative': np.float32(0.116736494), 'average_neutral': np.float32(0.20882826), 'average_positive': np.float32(0.6744353)}
    },
    'Brokeback Mountain': {
        'question1': {'average_negative': np.float32(0.8355453), 'average_neutral': np.float32(0.13083369), 'average_positive': np.float32(0.033620976)},
        'question2': {'average_negative': np.float32(0.107072204), 'average_neutral': np.float32(0.24898185), 'average_positive': np.float32(0.64394593)},
        'question3': {'average_negative': np.float32(0.1918805), 'average_neutral': np.float32(0.27968743), 'average_positive': np.float32(0.5284321)}
    },
    'Avatar': {
        'question1': {'average_negative': np.float32(0.7490596), 'average_neutral': np.float32(0.19352695), 'average_positive': np.float32(0.057413492)},
        'question2': {'average_negative': np.float32(0.07176623), 'average_neutral': np.float32(0.17062655), 'average_positive': np.float32(0.7576073)},
        'question3': {'average_negative': np.float32(0.19280389), 'average_neutral': np.float32(0.23368837), 'average_positive': np.float32(0.57350785)}
    },
    'Titanic': {
        'question1': {'average_negative': np.float32(0.68093854), 'average_neutral': np.float32(0.21729054), 'average_positive': np.float32(0.10177088)},
        'question2': {'average_negative': np.float32(0.13805524), 'average_neutral': np.float32(0.22205406), 'average_positive': np.float32(0.6398906)},
        'question3': {'average_negative': np.float32(0.22716871), 'average_neutral': np.float32(0.27825367), 'average_positive': np.float32(0.49457768)}
    },
    'Crouching Tiger, Hidden Dragon': {
        'question1': {'average_negative': np.float32(0.78773755), 'average_neutral': np.float32(0.1672009), 'average_positive': np.float32(0.045061592)},
        'question2': {'average_negative': np.float32(0.034463305), 'average_neutral': np.float32(0.15542588), 'average_positive': np.float32(0.81011087)},
        'question3': {'average_negative': np.float32(0.22207622), 'average_neutral': np.float32(0.25312632), 'average_positive': np.float32(0.5247974)}
    },
    'Nomadland': {
        'question1': {'average_negative': np.float32(0.7373511), 'average_neutral': np.float32(0.19154954), 'average_positive': np.float32(0.071099326)},
        'question2': {'average_negative': np.float32(0.082466215), 'average_neutral': np.float32(0.2226036), 'average_positive': np.float32(0.69493014)},
        'question3': {'average_negative': np.float32(0.1581859), 'average_neutral': np.float32(0.2817259), 'average_positive': np.float32(0.5600882)}
    }
}

# Convert the nested dictionary to a DataFrame
df = pd.DataFrame.from_dict({(i, j): data[i][j] 
                             for i in data.keys() 
                             for j in data[i].keys()},
                            orient='index')

# Reset the index to have movie and question as columns
df.reset_index(inplace=True)
df.columns = ['Movie', 'Question', 'Average Negative', 'Average Neutral', 'Average Positive']

# Save the DataFrame to a CSV file
df.to_csv('average_polarity_scores.csv', index=False)

print("DataFrame saved to 'average_polarity_scores.csv'")

