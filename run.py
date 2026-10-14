
import setproctitle
from dacite import from_dict

from common_ml.tagging.run_helpers import run_default, get_params, catch_errors

from ocr.ocr_model import OCRModel, RuntimeConfig
from config import config
        
if __name__ == '__main__':
    catch_errors()
    setproctitle.setproctitle("model-ocr")
    
    params = get_params()

    params = from_dict(RuntimeConfig, params)
    
    model = OCRModel(config["model_path"], runtime_config=params)

    run_default(model)