import pandas as pd
from surprise import Reader, Dataset, KNNBasic, SVD, accuracy, KNNWithMeans, KNNWithZScore, SVDpp, NMF
from surprise.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

# 1. Data Loading and Exploration
book_ratings = pd.read_csv('goodreads_ratings.csv')
print(book_ratings.head())

# Print dataset size and examine column data types
print(book_ratings.shape)
print(book_ratings.info())

# Distribution of ratings
print(book_ratings.rating.value_counts(normalize=True))

# 2. Data Preparation
# Filter ratings that are out of range
book_ratings = book_ratings[book_ratings['rating'] != 0]

# Prepare data for surprise: build a Surprise reader object
reader = Reader(rating_scale=(1, 5))

# Load book_ratings into a Surprise Dataset
data = Dataset.load_from_df(book_ratings[["user_id", "book_id", "rating"]], reader)

# Create a 80:20 train-test split and set the random state to 7
trainset, testset = train_test_split(data, test_size=.2, random_state=7)

# 3. Basic KNNBasic Model
print("\n--- Basic KNNBasic Model ---")
model = KNNBasic()
model.fit(trainset)

# Evaluate the recommender system
train_predictions = model.test(trainset.build_testset())
test_predictions = model.test(testset)

print(f'RMSE for training set: {accuracy.rmse(train_predictions):.4f}.')
print(f'RMSE for test set: {accuracy.rmse(test_predictions):.4f}.')

# Make a specific prediction
user_id = "8842281e1d1347389f2ab93d60773d4d"
book_id = 18007564
prediction = model.predict(user_id, book_id)

print(f"\nPredicted rating for 'The Martian': {prediction.est:.2f}")

# 4. KNNBasic Hyperparameter Tuning
print("\n--- KNNBasic Hyperparameter Tuning ---")
k_values = range(1, 51)
train_errors = []
test_errors = []
for k in k_values:
  model = KNNBasic(k=k)
  model.fit(trainset)
  train_predictions = model.test(trainset.build_testset())
  test_predictions = model.test(testset)
  train_errors.append(accuracy.rmse(train_predictions))
  test_errors.append(accuracy.rmse(test_predictions))

best_rmse_test = np.min(test_errors)
best_rmse_train = train_errors[np.argmin(test_errors)]
best_k = k_values[np.argmin(test_errors)]

print(f'Best model has k={best_k}, with train RMSE of {best_rmse_train:.4F} and test RMSE of {best_rmse_test:.4f}')

# 5. Testing different similarity metrics for KNNBasic
print("\n--- Testing Different Similarity Metrics ---")
sim_options = ['cosine', 'msd', 'pearson']
train_errors = []
test_errors = []

for sim in sim_options:
    model = KNNBasic(k=best_k, sim_options={'name': sim, 'user_based': True})
    model.fit(trainset)
    train_predictions = model.test(trainset.build_testset())
    test_predictions = model.test(testset)
    train_errors.append(accuracy.rmse(train_predictions))
    test_errors.append(accuracy.rmse(test_predictions))

# Print results
for i, sim in enumerate(sim_options):
    print(f'Similarity: {sim}, Train RMSE: {train_errors[i]:.4f}, Test RMSE: {test_errors[i]:.4f}')

# 6. Basic SVD Model
print("\n--- Basic SVD Model ---")
model = SVD()
model.fit(trainset)

train_predictions = model.test(trainset.build_testset())
test_predictions = model.test(testset)

print(f'RMSE for training set: {accuracy.rmse(train_predictions):.4f}.')
print(f'RMSE for test set: {accuracy.rmse(test_predictions):.4f}.')

# 7. SVD Epochs Tuning
print("\n--- SVD Epochs Tuning ---")
num_epochs = range(1, 51)
train_errors = []
test_errors = []
for n_epochs in num_epochs:
  model = SVD(n_epochs=n_epochs)
  model.fit(trainset)

  train_predictions = model.test(trainset.build_testset())
  test_predictions = model.test(testset)
  train_errors.append(accuracy.rmse(train_predictions))
  test_errors.append(accuracy.rmse(test_predictions))

best_rmse_test = np.min(test_errors)
best_rmse_train = train_errors[np.argmin(test_errors)]
best_num_epochs = num_epochs[np.argmin(test_errors)]

print(f'Best model has num_epochs={best_num_epochs}, with train RMSE of {best_rmse_train:.4F} and test RMSE of {best_rmse_test:.4f}')

plt.figure()
plt.plot(num_epochs, train_errors, color='r', label='Train')
plt.plot(num_epochs, test_errors, color='g', label='Test')
plt.xlabel('Number of Epochs')
plt.ylabel('RMSE')
plt.title('Epochs vs. RMSE')
plt.legend()
plt.show()
plt.close('all')

# 8. SVD Learning Rate Tuning
print("\n--- SVD Learning Rate Tuning ---")
lrs = np.logspace(-5, -1, 20)
train_errors = []
test_errors = []
for lr in lrs:
  model = SVD(n_epochs=best_num_epochs, lr_all=lr)
  model.fit(trainset)

  train_predictions = model.test(trainset.build_testset())
  test_predictions = model.test(testset)
  train_errors.append(accuracy.rmse(train_predictions))
  test_errors.append(accuracy.rmse(test_predictions))

best_rmse_test = np.min(test_errors)
best_rmse_train = train_errors[np.argmin(test_errors)]
best_lr = lrs[np.argmin(test_errors)]

print(f'Best model has num_epochs={best_num_epochs}, lr_all={best_lr:.2E}, with train RMSE of {best_rmse_train:.4F} and test RMSE of {best_rmse_test:.4f}')

plt.figure()
plt.plot(lrs, train_errors, color='r', label='Train')
plt.plot(lrs, test_errors, color='g', label='Test')
plt.xlabel('Learning Rate')
plt.ylabel('RMSE')
plt.title('Learning Rate vs RMSE')
plt.legend()
plt.show()
plt.close('all')

# 9. SVD Regularization Tuning
print("\n--- SVD Regularization Tuning ---")
regs = [0, 0.0002, 0.002, 0.008, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
train_errors = []
test_errors = []
for reg in regs:
  model = SVD(n_epochs=best_num_epochs, lr_all=best_lr, reg_all=reg)
  model.fit(trainset)

  train_predictions = model.test(trainset.build_testset())
  test_predictions = model.test(testset)
  train_errors.append(accuracy.rmse(train_predictions))
  test_errors.append(accuracy.rmse(test_predictions))

best_rmse_test = np.min(test_errors)
best_rmse_train = train_errors[np.argmin(test_errors)]
best_reg = regs[np.argmin(test_errors)]

print(f'Best model has num_epochs={best_num_epochs}, lr_all={best_lr:.2E}, reg={best_reg:.4f} with train RMSE of {best_rmse_train:.4F} and test RMSE of {best_rmse_test:.4f}')

plt.figure()
plt.plot(regs, train_errors, color='r', label='Train')
plt.plot(regs, test_errors, color='g', label='Test')
plt.xlabel('Regularization coefficient')
plt.ylabel('RMSE')
plt.title('Regularization vs. RMSE')
plt.legend()
plt.show()
plt.close('all')

# 10. Try Multiple Algorithms
print("\n--- Testing Different Algorithms ---")
algorithms = {
    'KNNWithMeans': KNNWithMeans(k=best_k),
    'KNNWithZScore': KNNWithZScore(k=best_k),
    'SVDpp': SVDpp(n_epochs=best_num_epochs, lr_all=best_lr, reg_all=best_reg),
    'NMF': NMF(n_epochs=best_num_epochs)
}

for name, algo in algorithms.items():
    algo.fit(trainset)
    train_predictions = algo.test(trainset.build_testset())
    test_predictions = algo.test(testset)
    print(f'Algorithm: {name}')
    print(f'RMSE for training set: {accuracy.rmse(train_predictions):.4f}')
    print(f'RMSE for test set: {accuracy.rmse(test_predictions):.4f}\n')


print("\n--- Grid Search for SVD ---")
param_grid = {
    'n_epochs': [best_num_epochs-5, best_num_epochs, best_num_epochs+5],
    'lr_all': [best_lr/2, best_lr, best_lr*2],
    'reg_all': [best_reg/2, best_reg, best_reg*2]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
gs.fit(data)

print(f"Best parameters: {gs.best_params['rmse']}")
print(f"Best RMSE: {gs.best_score['rmse']:.4f}")

best_model = SVD(**gs.best_params['rmse'])
best_model.fit(trainset)
test_predictions = best_model.test(testset)
print(f'Final test RMSE: {accuracy.rmse(test_predictions):.4f}')

# 12. Ensemble Model
print("\n--- Ensemble Model ---")
# Train several models with best parameters
model_knn = KNNBasic(k=best_k)
model_svd = SVD(n_epochs=best_num_epochs, lr_all=best_lr, reg_all=best_reg)
model_nmf = NMF(n_epochs=best_num_epochs)

# Fit all models
model_knn.fit(trainset)
model_svd.fit(trainset)
model_nmf.fit(trainset)

# Make predictions on test set
preds_knn = np.array([model_knn.predict(uid, iid).est for uid, iid, _ in testset])
preds_svd = np.array([model_svd.predict(uid, iid).est for uid, iid, _ in testset])
preds_nmf = np.array([model_nmf.predict(uid, iid).est for uid, iid, _ in testset])

# Try different weightings for ensemble
weights = [(0.2, 0.6, 0.2), (0.3, 0.4, 0.3), (0.4, 0.4, 0.2)]
actual = np.array([r for _, _, r in testset])

for w_knn, w_svd, w_nmf in weights:
    ensemble_preds = w_knn*preds_knn + w_svd*preds_svd + w_nmf*preds_nmf
    rmse = np.sqrt(np.mean((ensemble_preds - actual) ** 2))
    print(f'Ensemble weights (KNN, SVD, NMF): ({w_knn}, {w_svd}, {w_nmf}), RMSE: {rmse:.4f}')

# 13. Final Best Model Selection
print("\n--- Final Best Model ---")
# Use the best model configuration found across all experiments
best_model = SVD(n_epochs=best_num_epochs, lr_all=best_lr, reg_all=best_reg)
best_model.fit(trainset)

# Make prediction for the given user and book
final_prediction = best_model.predict(user_id, book_id)
print(f"Final predicted rating for 'The Martian': {final_prediction.est:.2f}")

# Get final performance metrics
test_predictions = best_model.test(testset)
final_rmse = accuracy.rmse(test_predictions)
print(f"Final model test RMSE: {final_rmse:.4f}")