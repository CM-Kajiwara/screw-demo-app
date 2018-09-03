import json
import logging
import os
from sagemaker.predictor import RealTimePredictor
from sagemaker.predictor import json_deserializer
from screw_predictor import ScrewPredictor
import base64
# endpoint = 'object-detection-2018-08-14-01-19-44-959'
endpoint = os.getenv('OBJECT_DETECTION_ENDPOINT','object-detection-2018-08-14-01-19-44-959')
object_predictor = ScrewPredictor(endpoint,content_type='image/jpeg',deserializer=json_deserializer)
threshold = 0.4
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def list_screw_detections_handler(event, context):
    logger.info(json.dumps(event))
    result = object_predictor.predict(base64.b64decode(event['body']),thresh=threshold)
    logger.info(json.dumps(result))
    response = {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(result)
    }
    logger.info(response)

    return response