"""
Kerala Super League Data Processor
==================================
A comprehensive Python script to convert JSON player data to CSV format
with advanced data cleaning, analysis, and insight generation.

Author: Rohit
Date: December 2024
Purpose: Showcase data analysis skills for Kerala Super League project
"""

import json
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class KeralaLeagueDataProcessor:
    """
    Comprehensive data processor for Kerala Super League player statistics
    """
    
    def __init__(self, input_file='/Users/rohit/kerala-super-league-data-analysis/data/raw/player_data.json', output_dir='/Users/rohit/kerala-super-league-data-analysis/data/processed/'):
        """
        Initialize the data processor
        
        Args:
            input_file (str): Path to input JSON file
            output_dir (str): Directory to save processed CSV files
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.raw_data = None
        self.processed_data = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"🏆 Kerala League Data Processor Initialized")
        logger.info(f"📥 Input: {input_file}")
        logger.info(f"📤 Output: {output_dir}")
    
    def load_json_data(self):
        """
        Load and validate JSON data from file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.input_file):
                logger.error(f"❌ File not found: {self.input_file}")
                logger.info("💡 Please place your JSON file in data/raw/player_data.json")
                return False
            
            with open(self.input_file, 'r', encoding='utf-8') as file:
                self.raw_data = json.load(file)
            
            logger.info(f"✅ Successfully loaded {len(self.raw_data)} player records")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON format: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
    
    def flatten_player_data(self):
        """
        Convert nested JSON structure to flat dictionary format
        
        Returns:
            list: List of flattened player dictionaries
        """
        logger.info("🔄 Flattening nested JSON structure...")
        
        flattened_players = []
        
        for i, player in enumerate(self.raw_data):
            try:
                # Extract main player information
                flattened_player = {
                    'player_id': player.get('player_id'),
                    'name': str(player.get('correct_player_name', '')).strip(),
                    'short_name': str(player.get('short_name', '')).strip(),
                    'age': player.get('age', 0),
                    'jersey_number': player.get('jersey_number'),
                    'position': str(player.get('position_actual', '')).strip(),
                    'is_foreign_player': player.get('is_foreign_player', 0),
                    'country': str(player.get('country', 'India')).strip(),
                    'team_name': str(player.get('team_name', '')).strip(),
                    'team_short_name': str(player.get('team_short_name', '')).strip(),
                    'photo_url': player.get('photo', ''),
                    'team_img_url': player.get('team_img_url', ''),
                }
                
                # Extract statistics from nested 'stats' object
                stats = player.get('stats', {})
                
                # Map all available statistics
                stat_mapping = {
                    'minutes_played': 'Minutes',
                    'goals': 'Goals',
                    'penalties': 'Penalties',
                    'chances_created': 'Chances Created',
                    'tackles': 'Tackles',
                    'interceptions': 'Interceptions',
                    'fouls': 'Fouls',
                    'fouls_suffered': 'Fouls Suffered',
                    'yellow_cards': 'Yellow Cards',
                    'red_cards': 'Red Cards',
                    'shots': 'Shots',
                    'shots_on_target': 'Shots on Target',
                    'passes': 'Passes',
                    'passes_completed': 'Passes completed',
                    'crosses': 'Crosses',
                    'short_passes': 'Short Passes',
                    'medium_passes': 'Medium Passes',
                    'long_passes': 'Long Passes',
                    'dribbles': 'Dribbles',
                    'clearances': 'Clearances',
                    'offsides': 'Offsides',
                    'assists': 'Assists',
                    'matches_played': 'Match Played',
                    'saves': 'Saves',
                    'goal_kicks': 'Goal Kick',
                    'goals_conceded': 'Goals conceded',
                    'penalties_saved': 'Penalties saved',
                    'penalties_conceded': 'Penalty conceded',
                    'passing_accuracy': 'Passing Accuracy'
                }
                
                # Add all statistics to flattened player data
                for new_key, original_key in stat_mapping.items():
                    value = stats.get(original_key, 0)
                    # Convert to numeric, handling any string values
                    try:
                        flattened_player[new_key] = float(value) if value != '' else 0
                    except (ValueError, TypeError):
                        flattened_player[new_key] = 0
                
                flattened_players.append(flattened_player)
                
            except Exception as e:
                logger.warning(f"⚠️  Error processing player {i}: {e}")
                continue
        
        logger.info(f"✅ Successfully flattened {len(flattened_players)} player records")
        return flattened_players
    
    def clean_and_enhance_data(self, data):
        """
        Clean data and calculate advanced metrics
        
        Args:
            data (list): List of flattened player dictionaries
            
        Returns:
            pandas.DataFrame: Cleaned and enhanced dataframe
        """
        logger.info("🧹 Cleaning and enhancing data...")
        
        df = pd.DataFrame(data)
        
        # Data cleaning
        # Handle missing or invalid ages
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['age'] = df['age'].replace(0, np.nan)
        
        # Clean text fields
        text_columns = ['name', 'team_name', 'position', 'country']
        for col in text_columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', '')
        
        # Ensure numeric columns are properly typed
        numeric_columns = [
            'minutes_played', 'goals', 'assists', 'tackles', 'interceptions',
            'passes', 'passes_completed', 'shots', 'shots_on_target',
            'matches_played', 'saves', 'goals_conceded', 'passing_accuracy'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calculate enhanced metrics
        logger.info("📊 Calculating advanced performance metrics...")
        
        # Per-match statistics
        df['goals_per_match'] = np.where(df['matches_played'] > 0, 
                                        df['goals'] / df['matches_played'], 0)
        
        df['assists_per_match'] = np.where(df['matches_played'] > 0, 
                                          df['assists'] / df['matches_played'], 0)
        
        df['minutes_per_match'] = np.where(df['matches_played'] > 0, 
                                          df['minutes_played'] / df['matches_played'], 0)
        
        # Goal contributions
        df['goal_contributions'] = df['goals'] + df['assists']
        df['goal_contributions_per_match'] = np.where(df['matches_played'] > 0,
                                                     df['goal_contributions'] / df['matches_played'], 0)
        
        # Defensive metrics
        df['defensive_actions'] = df['tackles'] + df['interceptions']
        df['defensive_actions_per_match'] = np.where(df['matches_played'] > 0,
                                                    df['defensive_actions'] / df['matches_played'], 0)
        
        # Shooting efficiency
        df['shooting_accuracy'] = np.where(df['shots'] > 0, 
                                          (df['shots_on_target'] / df['shots']) * 100, 0)
        
        # Passing efficiency
        df['passing_accuracy'] = np.where(df['passes'] > 0,
                                         (df['passes_completed'] / df['passes']) * 100,
                                         df['passing_accuracy'])
        
        # Position categorization
        def categorize_position(position):
            if pd.isna(position) or position == '':
                return 'Unknown'
            position = str(position).upper()
            if any(keyword in position for keyword in ['GK', 'GOALKEEPER', 'KEEPER']):
                return 'Goalkeeper'
            elif any(keyword in position for keyword in ['DEF', 'DEFENDER', 'BACK', 'CB', 'LB', 'RB']):
                return 'Defender'
            elif any(keyword in position for keyword in ['MID', 'MIDFIELDER']):
                return 'Midfielder'
            elif any(keyword in position for keyword in ['FOR', 'FORWARD', 'STRIKER', 'ATTACK', 'WING']):
                return 'Forward'
            else:
                return 'Midfielder'  # Default for unclear positions
        
        df['position_category'] = df['position'].apply(categorize_position)
        
        # Player efficiency rating (0-100 scale)
        def calculate_efficiency_rating(row):
            rating = 0
            
            # Goals component (0-30 points)
            if row['matches_played'] > 0:
                goals_per_match = row['goals'] / row['matches_played']
                rating += min(goals_per_match * 15, 30)
            
            # Assists component (0-20 points)
            if row['matches_played'] > 0:
                assists_per_match = row['assists'] / row['matches_played']
                rating += min(assists_per_match * 10, 20)
            
            # Passing accuracy component (0-25 points)
            if row['passing_accuracy'] > 0:
                rating += (row['passing_accuracy'] / 100) * 25
            
            # Defensive actions component (0-25 points)
            if row['matches_played'] > 0:
                def_per_match = row['defensive_actions'] / row['matches_played']
                rating += min(def_per_match * 5, 25)
            
            return min(rating, 100)
        
        df['efficiency_rating'] = df.apply(calculate_efficiency_rating, axis=1)
        
        # Data quality assessment
        df['data_quality_score'] = 0
        df.loc[df['age'].notna(), 'data_quality_score'] += 1
        df.loc[df['minutes_played'] > 0, 'data_quality_score'] += 1
        df.loc[df['matches_played'] > 0, 'data_quality_score'] += 1
        df.loc[df['position'] != '', 'data_quality_score'] += 1
        df.loc[df['team_name'] != '', 'data_quality_score'] += 1
        
        # Add processing timestamp
        df['data_processed_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"✅ Data cleaning completed. Final shape: {df.shape}")
        
        return df
    
    def generate_insights(self, df):
        """
        Generate comprehensive data insights and summary statistics
        
        Args:
            df (pandas.DataFrame): Processed dataframe
            
        Returns:
            dict: Summary insights and statistics
        """
        logger.info("🔍 Generating data insights...")
        
        insights = {
            'dataset_overview': {
                'total_players': len(df),
                'total_teams': df['team_name'].nunique(),
                'total_goals': int(df['goals'].sum()),
                'total_assists': int(df['assists'].sum()),
                'total_matches_played': int(df['matches_played'].sum()),
                'foreign_players': int((df['is_foreign_player'] == 1).sum()),
                'avg_age': round(df['age'].mean(), 1) if df['age'].notna().any() else 0,
                'data_quality_avg': round(df['data_quality_score'].mean(), 2)
            },
            
            'position_analysis': {
                'distribution': df['position_category'].value_counts().to_dict(),
                'goals_by_position': df.groupby('position_category')['goals'].sum().to_dict(),
                'avg_efficiency_by_position': df.groupby('position_category')['efficiency_rating'].mean().round(2).to_dict()
            },
            
            'team_analysis': {
                'squad_sizes': df['team_name'].value_counts().to_dict(),
                'goals_by_team': df.groupby('team_name')['goals'].sum().to_dict(),
                'assists_by_team': df.groupby('team_name')['assists'].sum().to_dict(),
                'avg_efficiency_by_team': df.groupby('team_name')['efficiency_rating'].mean().round(2).to_dict()
            },
            
            'top_performers': {
                'top_scorers': df.nlargest(10, 'goals')[['name', 'team_name', 'goals', 'matches_played']].to_dict('records'),
                'top_assisters': df.nlargest(10, 'assists')[['name', 'team_name', 'assists', 'matches_played']].to_dict('records'),
                'most_efficient': df.nlargest(10, 'efficiency_rating')[['name', 'team_name', 'efficiency_rating', 'position_category']].to_dict('records')
            },
            
            'performance_metrics': {
                'avg_goals_per_player': round(df['goals'].mean(), 2),
                'avg_assists_per_player': round(df['assists'].mean(), 2),
                'avg_passing_accuracy': round(df['passing_accuracy'].mean(), 1),
                'avg_minutes_per_match': round(df['minutes_per_match'].mean(), 1),
                'players_with_goals': int((df['goals'] > 0).sum()),
                'players_with_assists': int((df['assists'] > 0).sum())
            }
        }
        
        return insights
    
    def save_to_csv(self, df, insights):
        """
        Save processed data to multiple CSV formats
        
        Args:
            df (pandas.DataFrame): Processed dataframe
            insights (dict): Generated insights
            
        Returns:
            dict: Paths to saved files
        """
        logger.info("💾 Saving processed data to CSV files...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = {}
        
        # Main comprehensive dataset
        main_filename = f"kerala_league_players_complete_{timestamp}.csv"
        main_filepath = os.path.join(self.output_dir, main_filename)
        df.to_csv(main_filepath, index=False, encoding='utf-8')
        saved_files['main_dataset'] = main_filepath
        logger.info(f"📄 Main dataset saved: {main_filename}")
        
        # Team-specific datasets
        team_dir = os.path.join(self.output_dir, 'by_team')
        os.makedirs(team_dir, exist_ok=True)
        
        for team in df['team_name'].unique():
            if pd.notna(team) and team.strip():
                team_df = df[df['team_name'] == team]
                safe_team_name = "".join(c for c in team if c.isalnum() or c in (' ', '_')).replace(' ', '_')
                team_filename = f"{safe_team_name}_{timestamp}.csv"
                team_filepath = os.path.join(team_dir, team_filename)
                team_df.to_csv(team_filepath, index=False, encoding='utf-8')
        
        # Position-specific datasets
        position_dir = os.path.join(self.output_dir, 'by_position')
        os.makedirs(position_dir, exist_ok=True)
        
        for position in df['position_category'].unique():
            if pd.notna(position) and position.strip():
                position_df = df[df['position_category'] == position]
                position_filename = f"{position.lower()}_{timestamp}.csv"
                position_filepath = os.path.join(position_dir, position_filename)
                position_df.to_csv(position_filepath, index=False, encoding='utf-8')
        
        # Top performers summary
        top_performers_df = pd.concat([
            df.nlargest(5, 'goals')[['name', 'team_name', 'position_category', 'goals', 'efficiency_rating']],
            df.nlargest(5, 'assists')[['name', 'team_name', 'position_category', 'assists', 'efficiency_rating']],
            df.nlargest(5, 'efficiency_rating')[['name', 'team_name', 'position_category', 'efficiency_rating', 'goals', 'assists']]
        ]).drop_duplicates()
        
        top_performers_filename = f"top_performers_{timestamp}.csv"
        top_performers_filepath = os.path.join(self.output_dir, top_performers_filename)
        top_performers_df.to_csv(top_performers_filepath, index=False, encoding='utf-8')
        saved_files['top_performers'] = top_performers_filepath
        
        # Save insights as JSON
        insights_filename = f"data_insights_{timestamp}.json"
        insights_filepath = os.path.join(self.output_dir, insights_filename)
        with open(insights_filepath, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)
        saved_files['insights'] = insights_filepath
        
        logger.info(f"✅ All files saved successfully in {self.output_dir}")
        return saved_files
    
    def generate_summary_report(self, insights, saved_files):
        """
        Generate a comprehensive summary report
        
        Args:
            insights (dict): Generated insights
            saved_files (dict): Paths to saved files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"analysis_summary_report_{timestamp}.txt"
        report_filepath = os.path.join(self.output_dir, report_filename)
        
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write("🏆 KERALA SUPER LEAGUE - DATA ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 65 + "\n\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Processed by: Kerala League Data Processor v1.0\n\n")
            
            # Dataset Overview
            f.write("📊 DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            overview = insights['dataset_overview']
            f.write(f"Total Players: {overview['total_players']}\n")
            f.write(f"Total Teams: {overview['total_teams']}\n")
            f.write(f"Foreign Players: {overview['foreign_players']}\n")
            f.write(f"Average Age: {overview['avg_age']} years\n")
            f.write(f"Data Quality Score: {overview['data_quality_avg']}/5\n\n")
            
            # Performance Statistics
            f.write("⚽ PERFORMANCE STATISTICS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total Goals Scored: {overview['total_goals']}\n")
            f.write(f"Total Assists: {overview['total_assists']}\n")
            f.write(f"Total Matches Played: {overview['total_matches_played']}\n")
            
            perf = insights['performance_metrics']
            f.write(f"Average Goals per Player: {perf['avg_goals_per_player']}\n")
            f.write(f"Average Assists per Player: {perf['avg_assists_per_player']}\n")
            f.write(f"Average Passing Accuracy: {perf['avg_passing_accuracy']}%\n")
            f.write(f"Players with Goals: {perf['players_with_goals']}\n")
            f.write(f"Players with Assists: {perf['players_with_assists']}\n\n")
            
            # Position Analysis
            f.write("📍 POSITION ANALYSIS\n")
            f.write("-" * 18 + "\n")
            for position, count in insights['position_analysis']['distribution'].items():
                goals = insights['position_analysis']['goals_by_position'].get(position, 0)
                efficiency = insights['position_analysis']['avg_efficiency_by_position'].get(position, 0)
                f.write(f"{position}: {count} players, {goals} goals, {efficiency} avg efficiency\n")
            
            f.write("\n🏟️ TEAM ANALYSIS\n")
            f.write("-" * 15 + "\n")
            for team, size in insights['team_analysis']['squad_sizes'].items():
                goals = insights['team_analysis']['goals_by_team'].get(team, 0)
                assists = insights['team_analysis']['assists_by_team'].get(team, 0)
                f.write(f"{team}: {size} players, {goals} goals, {assists} assists\n")
            
            # Top Performers
            f.write("\n🏆 TOP PERFORMERS\n")
            f.write("-" * 15 + "\n")
            f.write("Top 5 Goal Scorers:\n")
            for i, player in enumerate(insights['top_performers']['top_scorers'][:5], 1):
                f.write(f"  {i}. {player['name']} ({player['team_name']}): {player['goals']} goals\n")
            
            f.write("\nTop 5 Assist Providers:\n")
            for i, player in enumerate(insights['top_performers']['top_assisters'][:5], 1):
                f.write(f"  {i}. {player['name']} ({player['team_name']}): {player['assists']} assists\n")
            
            f.write("\nMost Efficient Players:\n")
            for i, player in enumerate(insights['top_performers']['most_efficient'][:5], 1):
                f.write(f"  {i}. {player['name']} ({player['team_name']}): {player['efficiency_rating']:.1f} rating\n")
            
            # Files Generated
            f.write(f"\n📁 FILES GENERATED\n")
            f.write("-" * 16 + "\n")
            for file_type, filepath in saved_files.items():
                f.write(f"{file_type}: {os.path.basename(filepath)}\n")
        
        logger.info(f"📋 Summary report saved: {report_filename}")
        return report_filepath
    
    def process_data(self):
        """
        Main method to execute the complete data processing pipeline
        
        Returns:
            tuple: (success, main_csv_path, insights)
        """
        try:
            logger.info("🚀 Starting Kerala Super League data processing pipeline...")
            
            # Step 1: Load JSON data
            if not self.load_json_data():
                return False, None, None
            
            # Step 2: Flatten nested structure
            flattened_data = self.flatten_player_data()
            if not flattened_data:
                logger.error("❌ No data to process")
                return False, None, None
            
            # Step 3: Clean and enhance data
            processed_df = self.clean_and_enhance_data(flattened_data)
            self.processed_data = processed_df
            
            # Step 4: Generate insights
            insights = self.generate_insights(processed_df)
            
            # Step 5: Save all outputs
            saved_files = self.save_to_csv(processed_df, insights)
            
            # Step 6: Generate summary report
            report_path = self.generate_summary_report(insights, saved_files)
            
            logger.info("🎉 Data processing pipeline completed successfully!")
            logger.info(f"📊 Processed {len(processed_df)} players from {insights['dataset_overview']['total_teams']} teams")
            logger.info(f"📄 Main CSV: {saved_files['main_dataset']}")
            logger.info(f"📋 Summary report: {report_path}")
            
            return True, saved_files['main_dataset'], insights
            
        except Exception as e:
            logger.error(f"❌ Error in data processing pipeline: {e}")
            return False, None, None

def main():
    """
    Main function to run the Kerala Super League data processor
    """
    print("🏆 KERALA SUPER LEAGUE DATA PROCESSOR")
    print("=" * 50)
    print("Converting JSON player data to structured CSV format")
    print("with comprehensive analysis and insights generation")
    print("=" * 50)
    
    # Initialize processor
    processor = KeralaLeagueDataProcessor()
    
    # Process data
    success, csv_path, insights = processor.process_data()
    
    if success:
        print("\n✅ DATA PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 45)
        print(f"📊 Players processed: {insights['dataset_overview']['total_players']}")
        print(f"🏟️ Teams analyzed: {insights['dataset_overview']['total_teams']}")
        print(f"⚽ Total goals: {insights['dataset_overview']['total_goals']}")
        print(f"🎯 Total assists: {insights['dataset_overview']['total_assists']}")
        print(f"🌍 Foreign players: {insights['dataset_overview']['foreign_players']}")
        print(f"📈 Average efficiency: {insights['performance_metrics']['avg_passing_accuracy']:.1f}% passing accuracy")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"   Main dataset: {os.path.basename(csv_path)}")
        print(f"   Location: data/processed/")
        print(f"   Additional: Team & position breakdowns, top performers, insights")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Review the generated CSV files")
        print(f"   2. Open the summary report for detailed insights")
        print(f"   3. Use the data for visualization and further analysis")
        print(f"   4. Push to GitHub to showcase your data skills!")
        
    else:
        print("\n❌ DATA PROCESSING FAILED!")
        print("Please check the error messages above and ensure:")
        print("1. Your JSON file is in data/raw/player_data.json")
        print("2. The JSON file has valid format")
        print("3. You have write permissions in the data/processed/ directory")

if __name__ == "__main__":
    main()