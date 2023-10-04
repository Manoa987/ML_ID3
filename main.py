
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tree import DecisionTree
from attribute import Attribute
from forest import *


def main():
    # Read the Excel file
    file_path = "./src/german_credit.csv"

    df = pd.read_csv(file_path)

    # Create functions to discretize continuous values
    def discretize_duration(duration):
        if duration < 12:
            return 'Short Term'
        elif duration < 36:
            return 'Medium Term'
        else:
            return 'Long Term'

    def discretize_amount(amount):
        if amount < 1000:
            return 'Small Amount'
        elif amount < 2000:
            return 'Medium Amount'
        elif amount < 5000:
            return 'Big Amount'
        else:
            return 'Huge Amount'

    def discretize_Age(age):
        if age < 20:
            return 'Young'
        elif age < 45:
            return 'Young Adult'
        elif age < 80:
            return 'Adult'
        else:
            return 'Old'


    # Apply the discretize functions to create new columns
    df['duration_discretized'] = df['Duration of Credit (month)'].apply(discretize_duration)
    df['amount_discretized'] = df['Credit Amount'].apply(discretize_amount)
    df['age_discretized'] = df['Age (years)'].apply(discretize_Age)

    # Drop the original non-discretized columns
    df.drop(columns=['Duration of Credit (month)', 'Credit Amount','Age (years)'], inplace=True)

    #setup attributes
    target_attribute = Attribute('Creditability',['0','1'])
    attributes=[]
    for col in df.columns:
        aux=df[col].unique().tolist()
        aux.sort()
        if not col == "Creditability":
            attributes.append(Attribute(col,aux))
    
    
    TREE=True
    FOREST=True
    FOREST_DEPTH=False
    TREE_DEPTH=False
    ITERATIONS=10
    MAX_DEPTH=12
    NUMBER_OF_TREES=13
    
    results=[]
    accuracy_list=[]
    precision_list=[]
    recall_list=[]
    false_positive_list=[]
    f1_score_list=[]
    if TREE:
        for _ in range(ITERATIONS):
            test_set,train_set=generate_test_train_set(df)
             
            print("Training entries: ", len(train_set))
            print("Testing entries: ",len(test_set))
            results.append(tree_method(test_set, train_set, target_attribute, attributes,100))
        for i in range(ITERATIONS):
            confusion=results[i]    
            TP = confusion[0][0]
            FP = confusion[1][0]
            FN = confusion[0][1]
            TN = confusion[1][1]

            # Calculate accuracy, precision, recall, false positive rate, and F1-score
            accuracy_list.append((TP + TN) / len(test_set))
            precision_list.append( TP / (TP + FP))
            recall_list.append( TP / (TP + FN))
            false_positive_list.append( FP / (FP + TN))
            f1_score_list.append ( (2 * (( TP / (TP + FP)) * ( TP / (TP + FN)))) / (( TP / (TP + FP)) + ( TP / (TP + FN))))

        graphing_metrics(accuracy_list, precision_list, recall_list, false_positive_list, f1_score_list)

    if FOREST:
        
        for _ in range(ITERATIONS):
            test_set,train_set=generate_test_train_set(df)
             
            print("Training entries: ", len(train_set))
            print("Testing entries: ",len(test_set))

            results.append(random_forest_wrapper(test_set, train_set, target_attribute, attributes))
        for i in range(ITERATIONS):
            confusion=results[i]    
            TP = confusion[0][0]
            FP = confusion[1][0]
            FN = confusion[0][1]
            TN = confusion[1][1]

            # Calculate accuracy, precision, recall, false positive rate, and F1-score
            accuracy_list.append((TP + TN) / len(test_set))
            precision_list.append( TP / (TP + FP))
            recall_list.append( TP / (TP + FN))
            false_positive_list.append( FP / (FP + TN))
            f1_score_list.append ( (2 * (( TP / (TP + FP)) * ( TP / (TP + FN)))) / (( TP / (TP + FP)) + ( TP / (TP + FN))))

        graphing_metrics(accuracy_list, precision_list, recall_list, false_positive_list, f1_score_list)

    
    if TREE_DEPTH:
        
        curve_data=[]
        for iterations in range(ITERATIONS):
            test_set,train_set=generate_test_train_set(df)
            curve_data.append([])
            curve_data[iterations].append([]) #test
            curve_data[iterations].append([]) #train

            for depth in range(1,MAX_DEPTH):
                 #execute tree and get results
                my_tree=DecisionTree(depth,1)
                my_tree.train(train_set,attributes,target_attribute)
                    
                
                predicted_cats = []
                true_cats = []
                for i, row in test_set.iterrows():
                    true_cats.append(row[target_attribute.label])
                    prediction = my_tree.evaluate(row)
                    predicted_cats.append(prediction)
                    # Calculate confusion matrix
                confusion = confusion_matrix(true_cats, predicted_cats, labels=[1,0])
                TP = confusion[0][0]
                FP = confusion[1][0]
                curve_data[iterations][0].append(TP / (TP + FP))
                
                #execute tree on training set
                predicted_cats = []
                true_cats = []
                for i, row in train_set.iterrows():
                    true_cats.append(row[target_attribute.label])
                    prediction = my_tree.evaluate(row)
                    predicted_cats.append(prediction)
                    # Calculate confusion matrix
                confusion = confusion_matrix(true_cats, predicted_cats, labels=[1,0])

                TP = confusion[0][0]
                FP = confusion[1][0]
                curve_data[iterations][1].append(TP / (TP + FP))
                print("Depth "+str(depth)+"complete out of "+str(MAX_DEPTH-1))

                avg_run_result_curve=[[],[]]
        for depth in range(0,MAX_DEPTH-1):
            running_tally_test=0
            running_tally_train=0
            for iter in range(ITERATIONS):
                running_tally_test+=curve_data[iter][0][depth]
                running_tally_train+=curve_data[iter][1][depth]

            running_tally_test/=ITERATIONS
            running_tally_train/=ITERATIONS
            avg_run_result_curve[0].append(running_tally_test)
            avg_run_result_curve[1].append(running_tally_train)


        plt.figure(figsize=(10, 8))
        plt.plot(range(1,MAX_DEPTH),avg_run_result_curve[0],label="Test Tree")
        plt.plot(range(1,MAX_DEPTH),avg_run_result_curve[1],label="Train Tree")
        plt.xlabel('Tree Depth')
        plt.ylabel('Presicion%')
        plt.title('Variable Nodes Simple Tree 10 Run Average')
        plt.legend()
        # plt.show()
        plt.savefig('SimpleTreeTrainvsTest.png')

    if FOREST_DEPTH:
        curve_data_forest=[]
        for iterations in range(ITERATIONS):
            test_set,train_set=generate_test_train_set(df)
            curve_data_forest.append([])
            curve_data_forest[iterations].append([]) #test
            curve_data_forest[iterations].append([]) #train

            for depth in range(1,MAX_DEPTH):
            # for trees in range(1,NUMBER_OF_TREES):
                # print(trees)
                 #execute tree and get results
                my_forest=random_forest(train_set,attributes,target_attribute,100,depth,0,NUMBER_OF_TREES)
                                
              
                predicted_cats = []
                true_cats = []
                for i, row in test_set.iterrows():
                    true_cats.append(row[target_attribute.label])
                    prediction = random_forest_evaluate(my_forest,row)
                    predicted_cats.append(prediction)
                    # Calculate confusion matrix
                confusion = confusion_matrix(true_cats, predicted_cats, labels=[1,0])
                TP = confusion[0][0]
                FP = confusion[1][0]
                curve_data_forest[iterations][0].append(TP / (TP + FP))
                
                #execute tree on training set
                predicted_cats = []
                true_cats = []
                for i, row in train_set.iterrows():
                    true_cats.append(row[target_attribute.label])
                    prediction = random_forest_evaluate(my_forest,row)
                    predicted_cats.append(prediction)
                    # Calculate confusion matrix
                confusion = confusion_matrix(true_cats, predicted_cats, labels=[1,0])

                TP = confusion[0][0]
                FP = confusion[1][0]
                curve_data_forest[iterations][1].append(TP / (TP + FP))
                # print("Depth "+str(depth)+" complete out of "+str(MAX_DEPTH-1))
        

    
        avg_run_result_curve_forest=[[],[]]
        for depth in range(0,MAX_DEPTH-1):
        # for depth in range(0,NUMBER_OF_TREES-1):

            running_tally_test=0
            running_tally_train=0
            for iter in range(ITERATIONS):
                running_tally_test+=curve_data_forest[iter][0][depth]
                running_tally_train+=curve_data_forest[iter][1][depth]

            running_tally_test/=ITERATIONS
            running_tally_train/=ITERATIONS
            avg_run_result_curve_forest[0].append(running_tally_test)
            avg_run_result_curve_forest[1].append(running_tally_train)


        plt.figure(figsize=(10, 8))
        plt.plot(range(1,MAX_DEPTH),avg_run_result_curve_forest[0],label="Test Tree")
        plt.plot(range(1,MAX_DEPTH),avg_run_result_curve_forest[1],label="Train Tree")
        plt.xlabel('Tree Depth')
        plt.ylabel('Presicion%')
        plt.title('Variable Nodes Forest N-15 10 Run Average')
        plt.legend()
        # plt.show()
        plt.savefig('SimpleForestTrainvsTest.png')

    if FOREST_DEPTH and TREE_DEPTH:

        plt.figure(figsize=(10, 8))
        plt.plot(range(1,MAX_DEPTH),avg_run_result_curve[0],label="Test Tree")
        plt.plot(range(1,MAX_DEPTH),avg_run_result_curve[1],label="Train Tree")
        plt.plot(range(1,MAX_DEPTH),avg_run_result_curve_forest[0],label="Test Forest")
        plt.plot(range(1,MAX_DEPTH),avg_run_result_curve_forest[1],label="Train Forest")
        plt.xlabel('Tree Depth')
        plt.ylabel('Presicion%')
        plt.title('Variable Nodes 10 Run Average')
        plt.legend()
        # plt.show()
        plt.savefig('Simple-TreeandForest-TrainvsTest.png')


        
               
               
            
        
        
            













    
  
def generate_test_train_set(df):
    # Partition dataset
    percentage = 20 # Modify as needed
    # Shuffle the DataFrame
    shuffled_df = df.sample(frac=1)
    # Calculate the number of rows for the first partition
    partition_size = int(len(shuffled_df) * percentage / 100)

    # Split the shuffled DataFrame into test and train partitions
    test_set = shuffled_df.iloc[:partition_size]
    train_set = shuffled_df.iloc[partition_size:]
    test_set.reset_index(drop=True, inplace=True)
    train_set.reset_index(drop=True, inplace=True)

    return test_set,train_set

def graphing_metrics(accuracy_list, precision_list, recall_list, false_positive_list, f1_score_list):
    plt.figure(figsize=(10, 8))
    plt.boxplot(accuracy_list)
    plt.ylabel("Accuracy%")
    plt.title("Accuracy Tree Method")
    plt.show()

        #Precision
    plt.figure(figsize=(10, 8))
    plt.boxplot(precision_list)
    plt.ylabel("Precision%")
    plt.title("Precision Tree Method")
    plt.show()


        #Recall
    plt.figure(figsize=(10, 8))
    plt.boxplot(recall_list)
    plt.ylabel("True Positive Rate%")
    plt.title("Recall Tree Method")
    plt.show()

        #False Positive
    plt.figure(figsize=(10, 8))
    plt.boxplot(false_positive_list)
    plt.ylabel("False Positive Rate%")
    plt.title("False Positive Rate Tree Method")
    plt.show()
    

        #F1
    plt.figure(figsize=(10, 8))
    plt.boxplot(f1_score_list)
    plt.ylabel("F1 Score")
    plt.title("F1 Score Tree Method")
    plt.show()

def random_forest_wrapper(test_set, train_set, target_attribute, attributes):
    my_forest=random_forest(train_set,attributes,target_attribute,100,4,0,16)
        # random_forest_evaluate(my_forest,test_set)
        
        # print("node number is:"+str(my_tree.get_node_amount()))

        # Initialize data for confusion matrix and ROC
    predicted_cats = []
    true_cats = []
    for i, row in test_set.iterrows():
        true_cats.append(row[target_attribute.label])
        prediction = random_forest_evaluate(my_forest,row)
        predicted_cats.append(prediction)
        # Calculate confusion matrix
    confusion = confusion_matrix(true_cats, predicted_cats, labels=[1,0])
    print(confusion)
        # Display the confusion matrix as a heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False,
    #                 xticklabels=['Accepted','Refused'], yticklabels=['Accepted','Refused'])
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.show()

    return confusion

def tree_method(test_set, train_set, target_attribute, attributes,max_depth):
    my_tree=DecisionTree(max_depth,1)
    my_tree.train(train_set,attributes,target_attribute)
        
    print("node number is:"+str(my_tree.get_node_amount()))

        # Initialize data for confusion matrix and ROC
    predicted_cats = []
    true_cats = []
    for i, row in test_set.iterrows():
        true_cats.append(row[target_attribute.label])
        prediction = my_tree.evaluate(row)
        predicted_cats.append(prediction)
        # Calculate confusion matrix
    confusion = confusion_matrix(true_cats, predicted_cats, labels=[1,0])
    print(confusion)
        # Display the confusion matrix as a heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False,
    #                 xticklabels=['Accepted','Refused'], yticklabels=['Accepted','Refused'])
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.show()

    return confusion


if __name__ == "__main__":
    # This block of code will be executed if the script is run directly
    main()