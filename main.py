# ============================================================================
# PROJET ML - PRÉDICTION EURUSD
# ============================================================================

from KNN_and_Linear import run_baseline_models
from arima_model import run_arima_model
from prophet_model import run_prophet_model
from lstm_model import run_lstm_model
from datetime import datetime

# ============================================================================
# EXÉCUTION DES MODÈLES BASELINE
# ============================================================================

if __name__ == "__main__":
    try:
        start_time = datetime.now()
        
        results = run_baseline_models()
        arima_results = run_arima_model()
        prophet_results = run_prophet_model()
        lstm_results = run_lstm_model()
        
        print("\n" + "="*60)
        print("📈 MÉTRIQUES FINALES")
        print("="*60)
        print(f"\nLinear Regression → RMSE: {results['lr_metrics']['RMSE']:.6f} | R²: {results['lr_metrics']['R²']:.4f}")
        print(f"KNN Regression    → RMSE: {results['knn_metrics']['RMSE']:.6f} | R²: {results['knn_metrics']['R²']:.4f}")
        if arima_results:
            print(f"ARIMA             → RMSE: {arima_results['metrics']['RMSE']:.6f} | R²: {arima_results['metrics']['R²']:.4f} | Accuracy: {arima_results['metrics']['Accuracy']:.2f}%")
        else:
            print("ARIMA             → non disponible (package manquant)")
        if prophet_results:
            print(f"Prophet           → RMSE: {prophet_results['metrics']['RMSE']:.6f} | R²: {prophet_results['metrics']['R²']:.4f} | Accuracy: {prophet_results['metrics']['Accuracy']:.2f}%")
        else:
            print("Prophet           → non disponible (package manquant)")
        if lstm_results:
            print(f"LSTM              → RMSE: {lstm_results['metrics']['RMSE']:.6f} | R²: {lstm_results['metrics']['R²']:.4f} | Accuracy: {lstm_results['metrics']['Accuracy']:.2f}%")
        else:
            print("LSTM              → non disponible (package manquant)")
        
        # Déterminer le meilleur modèle (par RMSE) parmi ceux disponibles
        best = ('Linear Regression', results['lr_metrics']['RMSE'])
        if results['knn_metrics']['RMSE'] < best[1]:
            best = ('KNN Regression', results['knn_metrics']['RMSE'])
        if arima_results and arima_results['metrics']['RMSE'] < best[1]:
            best = ('ARIMA', arima_results['metrics']['RMSE'])
        if prophet_results and prophet_results['metrics']['RMSE'] < best[1]:
            best = ('Prophet', prophet_results['metrics']['RMSE'])
        if lstm_results and lstm_results['metrics']['RMSE'] < best[1]:
            best = ('LSTM', lstm_results['metrics']['RMSE'])

        print(f"\n🏆 Meilleur modèle: {best[0]} (RMSE: {best[1]:.6f})")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\n⏱️  Durée d'exécution: {duration:.2f}s")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {str(e)}")
        print(f"Type d'erreur: {type(e).__name__}")
        import traceback
        print("\nTraceback complet:")
        traceback.print_exc()
        print("\n❌❌❌ PROGRAMME INTERROMPU ❌❌❌")
