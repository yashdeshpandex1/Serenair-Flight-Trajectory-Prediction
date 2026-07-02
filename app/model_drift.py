import logging
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from inference_utils import initialize_inference_engine
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from preprocessing.data_prep import prep_live_inference_data
from modelling.custom_loss_fn import HaversineLoss
from modelling.engine import evaluate
from torch.utils.data import TensorDataset, DataLoader
from modelling.scaler_utilities import get_unscaled
from workers.db_utils import fetch_and_integrate_data
import time
import pandas as pd
import torch


logger = logging.getLogger(__name__)


def get_data_for_drift_check(duration_seconds=60):
    logger.info(f"Starting {duration_seconds}s data collection window..")
    start_time = time.time()
    collected_data = []
    
    while time.time() - start_time < duration_seconds:
        try:
            logger.info("Attempting to fetch data...")
            from workers.db_utils import fetch_and_integrate_data
            
            df = fetch_and_integrate_data('global')
            logger.info(f"✓ Fetch successful, got {len(df) if not df.empty else 0} records")
            
            if not df.empty:
                collected_data.append(df)
                logger.info(f"Collected batch: {len(df)} records (elapsed: {int(time.time() - start_time)}s)")
            else:
                logger.info("⚠ Empty dataframe returned")
                
            time.sleep(5)
        except Exception as e:
            logger.warning(f"Error during data collection: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)
            continue
    
    logger.info(f"Collection window ended. Got {len(collected_data)} batches total")
    
    if collected_data:
        df_combined = pd.concat(collected_data, ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['icao24', 'timestamp'])
        logger.info(f"✓ Combined and deduplicated: {len(df_combined)} records")
        return df_combined
    else:
        logger.warning("No data collected..")
        return pd.DataFrame()
    

def check_model_drift(device='cpu', collection_duration=300):
    try:
        logger.info("Loading models for drift check...")
        model_ni, scaler_ni = initialize_inference_engine(task='next_instance')
        model_ntm, scaler_ntm = initialize_inference_engine(task='next_ten_mins')
        
        if model_ni is None or model_ntm is None:
            logger.error("Could not load models for drift check")
            return None
        
        model_ni.to(device)
        model_ntm.to(device)
        model_ni.eval()
        model_ntm.eval()
        
        logger.info(f"Collecting data for {collection_duration} seconds...")
        df_live = get_data_for_drift_check(collection_duration)
        
        if df_live.empty:
            logger.warning('No data for drift check detected.')
            return None
        
        target_mean_ni, target_scale_ni = get_unscaled(task='next_instance')
        target_mean_ntm, target_scale_ntm = get_unscaled(task='next_ten_mins')
        
        target_mean_ni = target_mean_ni.to(device)
        target_scale_ni = target_scale_ni.to(device)
        target_mean_ntm = target_mean_ntm.to(device)
        target_scale_ntm = target_scale_ntm.to(device)
        
        # Next Instance
        X_tensor_ni, _ = prep_live_inference_data(df_live, window_size=10, task='next_instance')
        error_ni = None
        
        if X_tensor_ni.nelement() > 0:
            dataset_ni = TensorDataset(X_tensor_ni)
            loader_ni = DataLoader(dataset_ni, batch_size=128)
            
            error_ni = evaluate(
                model=model_ni,
                val_loader=loader_ni,
                eval_criterion=HaversineLoss,
                target_mean=target_mean_ni,
                target_scale=target_scale_ni,
                device=device,
                task='next_instance'
            )
            
        # Next Ten Minutes
        X_tensor_ntm, _ = prep_live_inference_data(df_live, window_size=10, task='next_ten_mins')
        error_ntm = None
        if X_tensor_ntm.nelement() > 0:
            dataset_ntm = TensorDataset(X_tensor_ntm)
            loader_ntm = DataLoader(dataset_ntm, batch_size=128)
            
            error_ntm = evaluate(
                model=model_ntm,
                val_loader=loader_ntm,
                eval_criterion=HaversineLoss,
                target_mean=target_mean_ntm,
                target_scale=target_scale_ntm,
                device=device,
                task='next_ten_mins'
            )
            
        logger.info(f"Drift check - Next Instance: {error_ni}m, Next Ten Minutes: {error_ntm}m")
        
        drift_detected = False
        alerts = []
        
        if error_ni and error_ni > 400:
            drift_detected = True
            alerts.append(f"Next Instance: {error_ni:.0f}m (threshold: 400m)")
        
        if error_ntm and error_ntm > 800:
            drift_detected = True
            alerts.append(f"Next 10min: {error_ntm:.0f}m (threshold: 800m)")
            
        return {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': drift_detected,
            'error_ni': error_ni,
            'error_ntm': error_ntm,
            'alerts': alerts
        }
        
    except Exception as e:
        logger.exception(f"Drift detection failed: {e}")
        return None
    
    
def send_drift_alert(drift_result):
    sender_email = os.getenv('ALERT_EMAIL')
    sender_password = os.getenv('ALERT_PASSWORD')
    recipient_email = os.getenv('YOUR_EMAIL')
    
    if not all([sender_email, sender_password, recipient_email]):
        logger.warning("Email alerts not configured!")
        return
    
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Serenair Drift Alert - {drift_result['timestamp']}"
        
        body = f"""
SERENAIR MODEL DRIFT DETECTION

Timestamp: {drift_result['timestamp']}

Next Instance Error:  {drift_result['error_ni']:.2f}m (Threshold: 400m)
Next 10min Error:     {drift_result['error_ntm']:.2f}m (Threshold: 800m)

Alerts:
{chr(10).join(drift_result['alerts']) if drift_result['alerts'] else 'None'}

Visit the site: https://www.serenair.live
"""
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        logger.info(f"✓ Drift alert sent to {recipient_email}")
        
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")
        

def main():
    """Main drift detection routine"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("SERENAIR MODEL DRIFT CHECK")
    logger.info("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Run drift check
    result = check_model_drift(device=device, collection_duration=60)
    
    if result is None:
        logger.warning("⚠ Drift check returned None")
        logger.info("=" * 60)
        return 1
    
    # Log results
    logger.info(f"Timestamp: {result['timestamp']}")
    logger.info(f"Next Instance Error: {result['error_ni']:.2f}m" if result['error_ni'] else "Next Instance Error: N/A")
    logger.info(f"Next 10min Error: {result['error_ntm']:.2f}m" if result['error_ntm'] else "Next 10min Error: N/A")
    
    # Handle drift
    if result['drift_detected']:
        logger.warning("🚨 DRIFT DETECTED!")
        for alert in result['alerts']:
            logger.warning(f"   ⚠ {alert}")
        send_drift_alert(result)
    else:
        logger.info("✓ No drift detected")
    
    logger.info("=" * 60)
    return 0


if __name__ == '__main__':
    import sys
    exit_code = main()
    sys.exit(exit_code)

