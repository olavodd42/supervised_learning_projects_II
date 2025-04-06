# import codecademylib3_seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

# Task 1: Look at the features
print("Aaron Judge DataFrame columns:")
print(aaron_judge.columns)

# Task 2: Examine the description feature
print("\nUnique values in the description feature:")
print(aaron_judge.description.unique())

# Task 3: Look at ball/strike encoding
print("\nUnique values in the type feature:")
print(aaron_judge.type.unique())

# Task 4: Convert strikes and balls to numeric values
aaron_judge['type'] = aaron_judge['type'].map({'S': 1, 'B': 0})

# Task 5: Check the conversion
print("\nType column after conversion:")
print(aaron_judge['type'])

# Task 6: Look at pitch location data
print("\nPlate X coordinates sample:")
print(aaron_judge['plate_x'].head(10))

# Task 7: Remove NaN values
aaron_judge = aaron_judge.dropna(subset=['plate_x', 'plate_z', 'type'])

# Create a function to analyze players
def analyze_player(player_data, player_name):
    """Analyze the strike zone for a baseball player using SVM"""
    print(f"\n--- Analyzing {player_name}'s Strike Zone ---")
    
    # Make a copy to avoid modifying the original data
    player_df = player_data.copy()
    
    # Convert types if not already done
    if 'S' in player_df['type'].unique() or 'B' in player_df['type'].unique():
        player_df['type'] = player_df['type'].map({'S': 1, 'B': 0})
    
    # Remove NaNs
    player_df = player_df.dropna(subset=['plate_x', 'plate_z', 'type'])
    print(f"Dataset shape after removing NaNs: {player_df.shape}")
    
    # Task 8: Create a scatter plot of the pitches
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.scatter(
        player_df['plate_x'], 
        player_df['plate_z'],
        c=player_df['type'],
        cmap=plt.cm.coolwarm,
        alpha=0.25
    )
    plt.xlabel('Horizontal position (plate_x)')
    plt.ylabel('Vertical position (plate_z)')
    plt.title(f'{player_name} Strike Zone')
    
    # Task 9: Split into training and validation sets
    training_set, validation_set = train_test_split(player_df, random_state=1)
    print(f"Training set size: {training_set.shape[0]}")
    print(f"Validation set size: {validation_set.shape[0]}")
    
    # Task 10: Create an SVC with default parameters
    classifier = SVC(kernel='rbf')
    
    # Task 11: Fit the classifier
    classifier.fit(
        training_set[['plate_x', 'plate_z']],
        training_set['type']
    )
    
    # Task 13: Score the basic model
    basic_accuracy = classifier.score(
        validation_set[['plate_x', 'plate_z']], 
        validation_set['type']
    )
    print(f"Basic SVM accuracy: {basic_accuracy:.4f}")
    
    # Task 14: Try with high gamma and C values
    overfit_classifier = SVC(kernel='rbf', gamma=100, C=100)
    overfit_classifier.fit(
        training_set[['plate_x', 'plate_z']],
        training_set['type']
    )
    overfit_accuracy = overfit_classifier.score(
        validation_set[['plate_x', 'plate_z']], 
        validation_set['type']
    )
    print(f"Overfit SVM accuracy (gamma=100, C=100): {overfit_accuracy:.4f}")
    
    # Task 15: Find the best parameters
    print("\nSearching for best parameters...")
    best_accuracy = 0
    best_gamma = 0
    best_C = 0
    best_classifier = None
    
    # Define ranges for gamma and C
    gammas = [0.1, 0.5, 1, 5, 10, 50, 100]
    Cs = [0.1, 1, 10, 100]
    
    for gamma in gammas:
        for C in Cs:
            temp_classifier = SVC(kernel='rbf', gamma=gamma, C=C)
            temp_classifier.fit(
                training_set[['plate_x', 'plate_z']],
                training_set['type']
            )
            accuracy = temp_classifier.score(
                validation_set[['plate_x', 'plate_z']], 
                validation_set['type']
            )
            print(f"  gamma={gamma}, C={C}: accuracy={accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_gamma = gamma
                best_C = C
                best_classifier = temp_classifier
    
    print(f"\nBest parameters: gamma={best_gamma}, C={best_C}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    
    # Task 12: Visualize the SVM decision boundary
    ax.set_ylim(-2, 6)
    ax.set_xlim(-3, 3)
    draw_boundary(ax, best_classifier)
    
    plt.show()
    
    # Task 17: Try with additional features (optional)
    print("\nTrying SVM with additional features...")
    
    # First, see which features might be most correlated with the type
    if 'strikes' in player_df.columns and 'balls' in player_df.columns:
        # Create an extended feature set
        features = ['plate_x', 'plate_z', 'strikes', 'balls']
        player_df_ext = player_df.dropna(subset=features + ['type'])
        
        train_ext, valid_ext = train_test_split(player_df_ext, random_state=1)
        
        ext_classifier = SVC(kernel='rbf', gamma=best_gamma, C=best_C)
        ext_classifier.fit(train_ext[features], train_ext['type'])
        ext_accuracy = ext_classifier.score(valid_ext[features], valid_ext['type'])
        
        print(f"Extended features accuracy: {ext_accuracy:.4f}")
        print("Note: Can't visualize boundary with >2 features")

# Run the analysis for each player
print("\n==== AARON JUDGE ANALYSIS ====")
analyze_player(aaron_judge, "Aaron Judge")

print("\n==== JOSE ALTUVE ANALYSIS ====")
analyze_player(jose_altuve, "Jose Altuve")

print("\n==== DAVID ORTIZ ANALYSIS ====")
analyze_player(david_ortiz, "David Ortiz")

# Optional: Determine the correlations with the target
print("\n==== FEATURE CORRELATION ANALYSIS ====")
aaron_judge_clean = aaron_judge.dropna(subset=['type'])
numeric_columns = aaron_judge_clean.select_dtypes(include=[np.number]).columns
corr = aaron_judge_clean[numeric_columns].corr()['type'].sort_values(ascending=False)
print("Top 10 correlations with pitch type:")
print(corr.head(11))  # Including 'type' itself