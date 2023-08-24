from qwak.automations import Automation, ScheduledTrigger, QwakBuildDeploy,\
    BuildSpecifications, BuildMetric, ThresholdDirection, DeploymentSpecifications


auto_scale_config = AutoScalingConfig(min_replica_count=1,
                                      max_replica_count=4,
                                      polling_interval=30,
                                      cool_down_period=300,
                                      triggers=[
                                          AutoScalingPrometheusTrigger(
                                              query_spec=AutoScaleQuerySpec(
                                                  aggregation_type="max",
                                                  metric_type="latency",
                                                  time_period=4),
                                              threshold=60
                                          )
                                      ]
                                      )


test_automation = Automation(
    name="retrain_breast_cancer",
    model_id="cancer_detection",
    trigger=ScheduledTrigger(cron="0 0 * * 0"),
    action=QwakBuildDeploy(
        build_spec=BuildSpecifications(git_uri="https://github.com/org_id/repository_name.git#dir_1/dir_2",
                                       git_access_token_secret="token_secret_name",
                                       git_branch="main",
                                       main_dir="main",
                                       tags=["prod"],
                                       env_vars=["key1=val1", "key2=val2", "key3=val3"]),
        deployment_condition=BuildMetric(metric_name="f1_score",
                                         direction=ThresholdDirection.ABOVE,
                                         threshold="0.65"),
        deployment_spec=DeploymentSpecifications(number_of_pods=1,
                                                 cpu_fraction=2.0,
                                                 memory="2Gi",
                                                 variation_name="B")
    )
)