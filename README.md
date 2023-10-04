# German Credit Classification

## Dataset Description

The dataset `german_credit.csv` contains information about individuals who applied for bank loans in German banks in the year 1994. It consists of 20 variables:

- `creditability`: Whether the individual repaid the credit (1) or not (0).
- `account.balance`: Values 1, 2, 3, 4 indicating no account (1), low balance (2), or well-balanced (4).
- `duration.of.credit..month.`: Duration of the credit in months.
- `payment.status.of.previous.credit`: Values from 0 to 4, where 0 indicates no payment and 4 indicates full payment.
- `purpose`: Values from 0 to 10, representing the purpose of the loan (e.g., 0 for a car).
- `credit.amount`: Numeric variable indicating the loan amount.
- `Svalue.savings.stocks`: Savings in monetary terms, values 1 to 5 (1 = none, 2 ≤ 100, 3 (100, 500], 4 (500, 1000].
- `length.of.current.employment`: Values 1 to 5, representing employment status (e.g., 1 for unemployed).
- `instalment.per.cent`: Values 1 to 4, indicating the percentage of financing (e.g., 1 for >35%).
- `sex...marital.status`: Values 1 to 4, representing gender and marital status.
- `guarantors`: Values 1 to 3, indicating guarantor type (e.g., 1 for none).
- `duration.in.current.address`: Values 1 to 4.
- `most.valuable.available.asset`: Values 1 to 4 (e.g., 1 for none, 2 for car).
- `age..years`: Numeric variable indicating age.
- `concurrent.credits`: Values 1 to 3, indicating concurrent credits.
- `type.of.apartment`: Values 1 to 3 (e.g., 1 for free).
- `no.of.credits.at.this.bank`: Values 1 to 4.
- `occupation`: Values 1 to 4, representing occupation.
- `no.of.dependents`: Values 1 or 2.
- `telephone`: Values 1 or 2 (yes or no).
- `foreign.worker`: Values 1 or 2 (yes or no).

## Tasks

### Data Splitting

1. Randomly split the dataset into two parts: a training set and a test set.

### ID3 Algorithm

2. Implement the ID3 algorithm for classifying the data to determine whether a person will repay the credit or not. Utilize all variables and Shannon's entropy for Gain calculation.

### Random Forest Classification

3. Classify the data using the Random Forest method, considering all variables.

### Confusion Matrix

4. Construct the confusion matrix for each method using the test set. Compare the results.

### Precision Curves

5. Generate precision curves for the decision tree model in terms of the number of nodes for each case. Plot the precision on both the training and test sets as a function of the number of nodes.

## Usage

1. Ensure you have the required Python libraries installed.
2. Run the code files for data splitting, ID3 algorithm, Random Forest classification, confusion matrix calculation, and precision curve generation.

## Contributions

Contributions to this project are welcome. If you have any improvements or suggestions, feel free to contribute by opening issues or creating pull requests.
