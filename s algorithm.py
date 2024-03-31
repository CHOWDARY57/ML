import csv

def find_s_algorithm(data):
   
    hypothesis = data[0][:-1]  
    for instance in data:
        if instance[-1] == 'Yes':  
            for i in range(len(hypothesis)):
                if instance[i] != hypothesis[i]:
                    hypothesis[i] = '?' 
    return hypothesis

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append(row)
    return data

def main():
   
    training_data = read_csv('C:\\Users\\ajayk\\OneDrive\\Documents\\finds and ce.csv')

    
    hypothesis = find_s_algorithm(training_data)

   
    print("The most specific hypothesis is:", hypothesis)

if __name__ == "__main__":
    main()
