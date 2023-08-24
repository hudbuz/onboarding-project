from qwak.feature_store.data_sources import SnowflakeSource
from qwak.clients.secret_service import SecretServiceClient
from qwak.feature_store.entities.entity import Entity
import qwak.feature_store.feature_sets.read_policies as ReadPolicy
from qwak.feature_store.feature_sets import batch
from qwak.feature_store.feature_sets.transformations import SparkSqlTransformation
import datetime




secret_service = SecretServiceClient()


snowflake_source = SnowflakeSource(
    name='snowflake_breast_cancer_source',
    description='a snowflake table for breast cancer detection data',
    date_created_column='ETL_TIME',
    host=secret_service.get_secret("snowflake-host"),
    username_secret_name="snowflake-user", # use secret service
    password_secret_name="snowflake-password", # use secret service
    database='QWAK_DB',
    schema='BREAST_CANCER_DETECTION',
    warehouse='COMPUTE_WH',
    table="BREAST_CANCER_DETECTION_RAW"
)


user = Entity(
    name='ID',
    description='id of the breast cancer dataset',
)

@batch.feature_set(
    name="breast-cancer-ingestion",
    entity="ID",
    data_sources = {
        "snowflake_breast_cancer_source": ReadPolicy.NewOnly
    })
# @batch.backfill(start_date=datetime.datetime(year=2022, month=12, day=30))
def user_features():
    return SparkSqlTransformation(sql="""
        SELECT 
        id,
        radius_mean,
        texture_mean, 
        perimeter_mean, 
        area_mean, 
        smoothness_mean, 
        compactness_mean, 
        concavity_mean,
        symmetry_mean, 
        fractal_dimension_mean,
        diagnosis,
        ETL_TIME
        FROM snowflake_breast_cancer_source
    """)