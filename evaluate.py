from sklearn.metrics import f1_score
import pandas as pd

true_file = 'original_training_set_labels.csv'
pred_file = 'submission.csv'

true_data = pd.read_csv(true_file)
pred_data = pd.read_csv(pred_file)

true_y = []
pred_y = []

for i, row in true_data.iterrows():
    filename = row[0]
    class_id = row[1]

    p_row = pred_data.loc[pred_data['filename'] == filename]
    p_class = p_row['category'].values[0]
    
    true_y += [class_id]
    pred_y += [p_class]

f1_scores = f1_score(true_y, pred_y, average=None)
m_f1 = f1_scores.mean()
print('MakroF1:',m_f1)

correct = 0
total = 0

for i, row in true_data.iterrows():
    filename = row[0]
    class_id = row[1]

    p_row = pred_data.loc[pred_data['filename'] == filename]
    p_class = p_row['category'].values[0]
    
    if class_id == p_class:
        correct += 1
    total += 1
    
acc = correct / total
print('Accuracy:',acc)
print('-'*8)
print('Total score:', 0.5*m_f1 + 0.5*acc)
