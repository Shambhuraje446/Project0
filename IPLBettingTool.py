import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class IPLBettingTool:
    def __init__(self, data_path='matches.csv'):
        """Initialize the IPL betting tool with data"""
        self.data_path = data_path
        self.df = None
        self.model = None
        self.team_stats = {}
        self.venue_stats = {}
        self.encoders = {}
        self.feature_importance = None  # Initialize feature_importance attribute
        
        # Load data
        self.load_data()
        
        # Train model if needed
        model_path = 'ipl_model.pkl'
        encoders_path = 'ipl_encoders.pkl'
        feature_importance_path = 'ipl_feature_importance.pkl'  # Add path for feature importance
        
        if (os.path.exists(model_path) and os.path.exists(encoders_path) 
            and os.path.exists(feature_importance_path)):  # Check if feature importance exists
            self.load_model(model_path, encoders_path, feature_importance_path)
        else:
            self.prepare_data()
            self.train_model()
            self.save_model(model_path, encoders_path, feature_importance_path)
    
    def load_data(self):
        """Load and preprocess the IPL matches data"""
        print("Loading IPL match data...")
        self.df = pd.read_csv(self.data_path)
        
        # Fill missing values
        self.df['player_of_match'] = self.df['player_of_match'].fillna('Not Awarded')
        self.df['winner'] = self.df['winner'].fillna('No Result')
        
        # Create useful features
        self.df['toss_winner_is_match_winner'] = (self.df['toss_winner'] == self.df['winner']).astype(int)
        
        # Create a feature for teams batting first
        self.df['batting_first'] = np.where(
            ((self.df['toss_winner'] == self.df['team1']) & (self.df['toss_decision'] == 'bat')) |
            ((self.df['toss_winner'] == self.df['team2']) & (self.df['toss_decision'] == 'field')),
            self.df['team1'], self.df['team2']
        )
        
        # Add a feature for whether batting first team won
        self.df['batting_first_won'] = (self.df['batting_first'] == self.df['winner']).astype(int)
        
        print("Data loaded successfully.")
    
    def prepare_data(self):
        """Prepare data for modeling and analysis"""
        print("Preparing data for analysis...")
        
        # Calculate team statistics
        teams = set(self.df['team1'].tolist() + self.df['team2'].tolist())
        for team in teams:
            if pd.isna(team) or team == 'No Result':
                continue
            
            # Matches played
            team1_matches = self.df[self.df['team1'] == team].shape[0]
            team2_matches = self.df[self.df['team2'] == team].shape[0]
            total_matches = team1_matches + team2_matches
            
            # Matches won
            matches_won = self.df[self.df['winner'] == team].shape[0]
            
            # Win rate
            win_rate = matches_won / total_matches if total_matches > 0 else 0
            
            # Toss wins
            toss_wins = self.df[self.df['toss_winner'] == team].shape[0]
            toss_win_rate = toss_wins / total_matches if total_matches > 0 else 0
            
            # Last 5 matches form
            recent_matches = self.df[(self.df['team1'] == team) | (self.df['team2'] == team)].tail(5)
            recent_wins = recent_matches[recent_matches['winner'] == team].shape[0]
            recent_form = recent_wins / recent_matches.shape[0] if recent_matches.shape[0] > 0 else 0
            
            # Store team stats
            self.team_stats[team] = {
                'matches_played': total_matches,
                'matches_won': matches_won,
                'win_rate': win_rate,
                'toss_wins': toss_wins,
                'toss_win_rate': toss_win_rate,
                'recent_form': recent_form
            }
        
        # Calculate venue statistics
        if 'venue' in self.df.columns:
            for venue in self.df['venue'].unique():
                if pd.isna(venue):
                    continue
                    
                venue_matches = self.df[self.df['venue'] == venue]
                batting_first_wins = venue_matches['batting_first_won'].sum()
                total_venue_matches = venue_matches.shape[0]
                
                batting_first_win_rate = batting_first_wins / total_venue_matches if total_venue_matches > 0 else 0
                
                self.venue_stats[venue] = {
                    'matches_played': total_venue_matches,
                    'batting_first_win_rate': batting_first_win_rate
                }
        
        # Encode categorical features
        categorical_features = ['city', 'team1', 'team2', 'toss_winner', 'venue']
        for feature in categorical_features:
            if feature in self.df.columns:
                encoder = LabelEncoder()
                self.df[f'{feature}_encoded'] = encoder.fit_transform(self.df[feature].fillna('Unknown'))
                self.encoders[feature] = encoder
        
        print("Data preparation complete.")
    
    def train_model(self):
        """Train the predictive model"""
        print("Training model...")
        
        # Prepare features and target
        feature_cols = [col for col in self.df.columns if col.endswith('_encoded')]
        feature_cols += ['toss_winner_is_match_winner']
        
        # Keep only relevant rows
        model_df = self.df[self.df['winner'] != 'No Result'].copy()
        
        # Target: Will team1 win?
        model_df['team1_won'] = (model_df['team1'] == model_df['winner']).astype(int)
        
        # Remove rows with missing features
        model_df = model_df.dropna(subset=feature_cols)
        
        # Train model
        X = model_df[feature_cols]
        y = model_df['team1_won']
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Model training complete.")
    
    def save_model(self, model_path, encoders_path, feature_importance_path):
        """Save model, encoders, and feature importance to disk"""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.encoders, f)
        
        # Save feature importance
        with open(feature_importance_path, 'wb') as f:
            pickle.dump(self.feature_importance, f)
        
        print(f"Model and related data saved to disk")
    
    def load_model(self, model_path, encoders_path, feature_importance_path):
        """Load model, encoders, and feature importance from disk"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(encoders_path, 'rb') as f:
            self.encoders = pickle.load(f)
        
        # Load feature importance
        with open(feature_importance_path, 'rb') as f:
            self.feature_importance = pickle.load(f)
        
        self.prepare_data()  # Still need to calculate stats
        print(f"Model loaded from {model_path}")
    
    def predict_match(self, team1, team2, venue, toss_winner, toss_decision):
        """Predict the outcome of a match"""
        if self.model is None:
            print("Model not trained yet. Call train_model() first.")
            return None
        
        # Create feature vector
        features = {}
        
        # Check if teams and venue exist in encoders
        for entity, value in [('team1', team1), ('team2', team2), ('venue', venue), ('toss_winner', toss_winner)]:
            if entity in self.encoders:
                if value in self.encoders[entity].classes_:
                    features[f'{entity}_encoded'] = self.encoders[entity].transform([value])[0]
                else:
                    print(f"Warning: {value} not in training data for {entity}. Using default encoding.")
                    features[f'{entity}_encoded'] = 0
        
        # Add city encoding if available
        if 'city' in self.encoders:
            # Find the city for the venue
            venue_data = self.df[self.df['venue'] == venue]
            if not venue_data.empty and 'city' in venue_data.columns:
                city = venue_data['city'].iloc[0]
                if city in self.encoders['city'].classes_:
                    features['city_encoded'] = self.encoders['city'].transform([city])[0]
                else:
                    features['city_encoded'] = 0
            else:
                features['city_encoded'] = 0
        
        # Calculate toss winner is match winner probability from historical data
        toss_winner_data = self.df[self.df['toss_winner'] == toss_winner]
        if not toss_winner_data.empty:
            toss_winner_win_rate = toss_winner_data['toss_winner_is_match_winner'].mean()
        else:
            toss_winner_win_rate = 0.5  # Default if no historical data
            
        features['toss_winner_is_match_winner'] = toss_winner_win_rate
        
        # Get feature columns from feature importance
        feature_cols = self.feature_importance['Feature'].tolist()
        
        # Make prediction
        feature_values = [features.get(col, 0) for col in feature_cols]
        prediction = self.model.predict([feature_values])
        probability = self.model.predict_proba([feature_values])
        
        # Get win probabilities
        team1_win_prob = probability[0][1]
        team2_win_prob = 1 - team1_win_prob
        
        # Determine batting first team
        if (toss_winner == team1 and toss_decision == 'bat') or (toss_winner == team2 and toss_decision == 'field'):
            batting_first = team1
            batting_second = team2
        else:
            batting_first = team2
            batting_second = team1
            
        # Calculate venue advantage
        venue_advantage = None
        if venue in self.venue_stats:
            batting_first_win_rate = self.venue_stats[venue]['batting_first_win_rate']
            if batting_first_win_rate > 0.55:
                venue_advantage = f"Batting first has an advantage at {venue} ({batting_first_win_rate:.2f} win rate)"
                if batting_first == team1:
                    team1_win_prob = (team1_win_prob + batting_first_win_rate) / 2
                    team2_win_prob = 1 - team1_win_prob
                else:
                    team2_win_prob = (team2_win_prob + batting_first_win_rate) / 2
                    team1_win_prob = 1 - team2_win_prob
            elif batting_first_win_rate < 0.45:
                venue_advantage = f"Batting second has an advantage at {venue} ({(1-batting_first_win_rate):.2f} win rate)"
                if batting_second == team1:
                    team1_win_prob = (team1_win_prob + (1-batting_first_win_rate)) / 2
                    team2_win_prob = 1 - team1_win_prob
                else:
                    team2_win_prob = (team2_win_prob + (1-batting_first_win_rate)) / 2
                    team1_win_prob = 1 - team2_win_prob
        
        # Calculate recent form advantage
        form_advantage = None
        if team1 in self.team_stats and team2 in self.team_stats:
            team1_form = self.team_stats[team1].get('recent_form', 0)
            team2_form = self.team_stats[team2].get('recent_form', 0)
            if abs(team1_form - team2_form) > 0.2:  # Significant form difference
                if team1_form > team2_form:
                    form_advantage = f"{team1} is in better recent form ({team1_form:.2f} vs {team2_form:.2f})"
                    team1_win_prob = (team1_win_prob + team1_form) / 2
                    team2_win_prob = 1 - team1_win_prob
                else:
                    form_advantage = f"{team2} is in better recent form ({team2_form:.2f} vs {team1_form:.2f})"
                    team2_win_prob = (team2_win_prob + team2_form) / 2
                    team1_win_prob = 1 - team2_win_prob
        
        # Calculate head-to-head advantage
        h2h_advantage = None
        h2h_matches = self.df[((self.df['team1'] == team1) & (self.df['team2'] == team2)) | 
                              ((self.df['team1'] == team2) & (self.df['team2'] == team1))]
        
        if not h2h_matches.empty:
            team1_h2h_wins = h2h_matches[h2h_matches['winner'] == team1].shape[0]
            team2_h2h_wins = h2h_matches[h2h_matches['winner'] == team2].shape[0]
            total_h2h = h2h_matches.shape[0]
            
            if total_h2h > 0:
                team1_h2h_rate = team1_h2h_wins / total_h2h
                team2_h2h_rate = team2_h2h_wins / total_h2h
                
                if abs(team1_h2h_rate - team2_h2h_rate) > 0.15:  # Significant H2H difference
                    if team1_h2h_rate > team2_h2h_rate:
                        h2h_advantage = f"{team1} has a head-to-head advantage ({team1_h2h_rate:.2f} win rate in {total_h2h} matches)"
                        team1_win_prob = (team1_win_prob + team1_h2h_rate) / 2
                        team2_win_prob = 1 - team1_win_prob
                    else:
                        h2h_advantage = f"{team2} has a head-to-head advantage ({team2_h2h_rate:.2f} win rate in {total_h2h} matches)"
                        team2_win_prob = (team2_win_prob + team2_h2h_rate) / 2
                        team1_win_prob = 1 - team2_win_prob
        
        # Return prediction results
        result = {
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'batting_first': batting_first,
            'team1_win_probability': team1_win_prob,
            'team2_win_probability': team2_win_prob,
            'venue_advantage': venue_advantage,
            'form_advantage': form_advantage,
            'h2h_advantage': h2h_advantage
        }
        
        return result
    
    def display_prediction(self, prediction):
        """Display prediction results in a readable format"""
        if prediction is None:
            return
        
        print("\n" + "=" * 60)
        print(f"MATCH PREDICTION: {prediction['team1']} vs {prediction['team2']}")
        print("=" * 60)
        
        print(f"\nVenue: {prediction['venue']}")
        print(f"Toss: {prediction['toss_winner']} won and chose to {prediction['toss_decision']}")
        print(f"Batting first: {prediction['batting_first']}")
        
        print("\nWIN PROBABILITIES:")
        print(f"{prediction['team1']}: {prediction['team1_win_probability']:.2f} ({prediction['team1_win_probability']*100:.1f}%)")
        print(f"{prediction['team2']}: {prediction['team2_win_probability']:.2f} ({prediction['team2_win_probability']*100:.1f}%)")
        
        print("\nADVANTAGES:")
        if prediction['venue_advantage']:
            print(f"- Venue: {prediction['venue_advantage']}")
        if prediction['form_advantage']:
            print(f"- Form: {prediction['form_advantage']}")
        if prediction['h2h_advantage']:
            print(f"- H2H: {prediction['h2h_advantage']}")
        
        # Calculate betting odds and recommendations
        favorite = prediction['team1'] if prediction['team1_win_probability'] > prediction['team2_win_probability'] else prediction['team2']
        underdog = prediction['team2'] if favorite == prediction['team1'] else prediction['team1']
        
        fav_prob = max(prediction['team1_win_probability'], prediction['team2_win_probability'])
        dog_prob = min(prediction['team1_win_probability'], prediction['team2_win_probability'])
        
        # Convert probabilities to fair odds
        fav_fair_odds = 1 / fav_prob if fav_prob > 0 else float('inf')
        dog_fair_odds = 1 / dog_prob if dog_prob > 0 else float('inf')
        
        print("\nBETTING ANALYSIS:")
        print(f"Fair odds for {favorite}: {fav_fair_odds:.2f}")
        print(f"Fair odds for {underdog}: {dog_fair_odds:.2f}")
        
        print("\nBETTING RECOMMENDATIONS:")
        if fav_prob > 0.65:
            print(f"- STRONG BET on {favorite} if odds > {fav_fair_odds:.2f}")
        elif fav_prob > 0.55:
            print(f"- MODERATE BET on {favorite} if odds > {fav_fair_odds:.2f}")
        else:
            print(f"- AVOID betting on this match - outcome too unpredictable")
            
        print(f"- Value bet on {underdog} if odds > {dog_fair_odds+0.5:.2f}")
        
        # Confidence level
        confidence_factors = sum(x is not None for x in [prediction['venue_advantage'], 
                                                         prediction['form_advantage'], 
                                                         prediction['h2h_advantage']])
        confidence = "HIGH" if confidence_factors >= 2 else "MEDIUM" if confidence_factors == 1 else "LOW"
        
        print(f"\nPrediction confidence: {confidence}")
        print("=" * 60)
    
    def get_available_teams(self):
        """Get list of teams in the dataset"""
        teams = set(self.df['team1'].tolist() + self.df['team2'].tolist())
        return sorted([t for t in teams if pd.notna(t) and t != 'No Result'])
    
    def get_available_venues(self):
        """Get list of venues in the dataset"""
        return sorted([v for v in self.df['venue'].unique() if pd.notna(v)])
    
    def get_team_stats(self, team):
        """Get detailed statistics for a team"""
        if team not in self.team_stats:
            return None
        
        stats = self.team_stats[team].copy()
        
        # Add head-to-head records
        h2h_records = {}
        for opponent in self.get_available_teams():
            if opponent == team:
                continue
                
            h2h_matches = self.df[((self.df['team1'] == team) & (self.df['team2'] == opponent)) | 
                                  ((self.df['team1'] == opponent) & (self.df['team2'] == team))]
            
            team_wins = h2h_matches[h2h_matches['winner'] == team].shape[0]
            opponent_wins = h2h_matches[h2h_matches['winner'] == opponent].shape[0]
            total_matches = h2h_matches.shape[0]
            
            if total_matches > 0:
                h2h_records[opponent] = {
                    'matches': total_matches,
                    'wins': team_wins,
                    'losses': opponent_wins,
                    'win_rate': team_wins / total_matches
                }
        
        stats['h2h_records'] = h2h_records
        
        # Add venue performance
        venue_performance = {}
        for venue in self.get_available_venues():
            venue_matches = self.df[((self.df['team1'] == team) | (self.df['team2'] == team)) & 
                                    (self.df['venue'] == venue)]
            
            team_wins = venue_matches[venue_matches['winner'] == team].shape[0]
            total_matches = venue_matches.shape[0]
            
            if total_matches >= 3:  # Only include venues with enough matches
                venue_performance[venue] = {
                    'matches': total_matches,
                    'wins': team_wins,
                    'win_rate': team_wins / total_matches
                }
        
        stats['venue_performance'] = venue_performance
        
        return stats
    
    def display_team_stats(self, team):
        """Display detailed statistics for a team"""
        stats = self.get_team_stats(team)
        if stats is None:
            print(f"No data available for team: {team}")
            return
        
        print("\n" + "=" * 60)
        print(f"TEAM STATISTICS: {team}")
        print("=" * 60)
        
        print(f"\nOverall record: {stats['matches_won']} wins in {stats['matches_played']} matches")
        print(f"Win rate: {stats['win_rate']:.3f} ({stats['win_rate']*100:.1f}%)")
        print(f"Recent form: {stats['recent_form']:.3f} ({stats['recent_form']*100:.1f}%)")
        
        print("\nHead-to-head records:")
        for opponent, h2h in sorted(stats['h2h_records'].items(), 
                                    key=lambda x: x[1]['win_rate'], 
                                    reverse=True):
            print(f"- vs {opponent}: {h2h['wins']}-{h2h['losses']} ({h2h['win_rate']*100:.1f}%)")
        
        print("\nVenue performance:")
        for venue, perf in sorted(stats['venue_performance'].items(), 
                                 key=lambda x: x[1]['win_rate'], 
                                 reverse=True):
            print(f"- {venue}: {perf['wins']} wins in {perf['matches']} matches ({perf['win_rate']*100:.1f}%)")
            
        print("=" * 60)
    
    def interactive_prediction(self):
        """Run an interactive prediction session"""
        print("\nINTERACTIVE IPL MATCH PREDICTION")
        print("=" * 60)
        
        teams = self.get_available_teams()
        venues = self.get_available_venues()
        
        print("\nAvailable teams:")
        for i, team in enumerate(teams, 1):
            print(f"{i}. {team}")
        
        team1_idx = int(input("\nSelect team 1 (number): ")) - 1
        team1 = teams[team1_idx]
        
        team2_idx = int(input("Select team 2 (number): ")) - 1
        team2 = teams[team2_idx]
        
        print("\nAvailable venues:")
        for i, venue in enumerate(venues, 1):
            print(f"{i}. {venue}")
        
        venue_idx = int(input("\nSelect venue (number): ")) - 1
        venue = venues[venue_idx]
        
        toss_choice = input("\nWhich team won the toss? (1/2): ")
        toss_winner = team1 if toss_choice == '1' else team2
        
        toss_decision = input("Toss winner chose to (bat/field): ").lower()
        
        # Make prediction
        prediction = self.predict_match(team1, team2, venue, toss_winner, toss_decision)
        self.display_prediction(prediction)
        
        # Offer team stats
        stats_choice = input("\nDo you want to see detailed stats for any team? (1/2/n): ")
        if stats_choice == '1':
            self.display_team_stats(team1)
        elif stats_choice == '2':
            self.display_team_stats(team2)
            
        print("\nInteractive prediction complete.")
        return prediction

# Main execution
if __name__ == "__main__":
    print("IPL Betting Assistant Tool")
    print("=" * 60)
    print("This tool helps you make data-driven betting decisions for IPL matches.")
    print("It uses historical match data to predict outcomes and provide betting recommendations.")
    print("=" * 60)
    
    # Initialize betting tool
    betting_tool = IPLBettingTool()
    
    while True:
        print("\nMENU:")
        print("1. Predict match outcome")
        print("2. View team statistics")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ")
        
        if choice == '1':
            betting_tool.interactive_prediction()
        elif choice == '2':
            teams = betting_tool.get_available_teams()
            print("\nAvailable teams:")
            for i, team in enumerate(teams, 1):
                print(f"{i}. {team}")
            
            team_idx = int(input("\nSelect team (number): ")) - 1
            team = teams[team_idx]
            betting_tool.display_team_stats(team)
        elif choice == '3':
            print("\nThank you for using the IPL Betting Assistant Tool!")
            break
        else:
            print("Invalid choice. Please try again.")