

import pandas as pd

def candidate_elimination(training_data):
    specific_h = training_data.iloc[0, :-1].tolist()
    general_h = [['?' for _ in specific_h] for _ in specific_h]
    
    for _, instance in training_data.iterrows():
        print("Instance:", instance.tolist())
        if instance.iloc[-1] == 'Malignant(+)':
            specific_h = ['?' if specific_h[i] != val else val for i, val in enumerate(instance.iloc[:-1])]
            for i in range(len(specific_h)):
                if specific_h[i] != instance[i]:
                    general_h[i][i] = specific_h[i]
                else:
                    general_h[i][i] = '?'
        else:
            for i in range(len(specific_h)):
                if specific_h[i] == instance[i]:
                    general_h[i][i] = specific_h[i]
                else:
                    general_h[i][i] = '?'
        
        print("Specific Hypothesis:", specific_h)
        print("General Hypotheses:")
        for h in general_h:
            print(pd.Series(h))
        print("-------------------------------")
    
    general_h = [pd.Series(general) for general in general_h if '?' in general]
    return specific_h, general_h

def main():
    training_data = pd.read_csv('C:\\Users\\ajayk\\OneDrive\\Documents\\finds and ce.csv')

    specific_hypothesis, general_hypotheses = candidate_elimination(training_data)
    print("Most specific hypothesis:", specific_hypothesis)
    print("General hypotheses:")
    print("Specific Hypothesis:")
    print(specific_hypothesis)
    print("General Hypotheses:")
    for h in general_hypotheses:
        print(h)

if __name__ == "__main__":
    main()
