set -euo pipefail
cd "$(dirname "$0")/../.."

DATA_VERSION="${1:-unversioned}"
TRAIN="python src/models/train.py --data-version ${DATA_VERSION}"

$TRAIN --model dummy --run-name baseline-dummy
$TRAIN --model logistic_regression --c 1.0 --run-name lr-c1.0
$TRAIN --model logistic_regression --c 0.1 --class-weight balanced --run-name lr-c0.1-balanced
$TRAIN --model xgboost --n-estimators 100 --max-depth 3 --learning-rate 0.1 --run-name xgb-100-d3-lr0.1
$TRAIN --model xgboost --n-estimators 200 --max-depth 4 --learning-rate 0.05 --run-name xgb-200-d4-lr0.05
$TRAIN --model xgboost --n-estimators 300 --max-depth 6 --learning-rate 0.03 --class-weight balanced --run-name xgb-300-d6-lr0.03-balanced