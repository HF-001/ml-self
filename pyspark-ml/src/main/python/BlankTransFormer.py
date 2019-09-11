"""
自定义transformer，方便进行数据预处理，将自定义的tansformer加入pipleline,
通过fit(),transform()进行调用。
"""
from pyspark.sql import SparkSession

#构建SparkSession
spark = SparkSession \
    .builder \
    .appName("test") \
    .master("local") \
    .getOrCreate()

from pyspark import keyword_only  ##
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCols, Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

# 缺失值补0
class BlankTransformer(Transformer, HasInputCols, HasOutputCols, DefaultParamsReadable, DefaultParamsWritable):

    @keyword_only
    def __init__(self, inputCols = None, outputCols = None):
        super(BlankTransformer, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @keyword_only
    def setParams(self, outputCols=None, value=0.0):
        """
        setParams(self, outputCols=None, value=0.0)
        Sets params for this SetValueTransformer.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setValue(self, value):
        """
        Sets the value of :py:attr:`value`.
        """
        return self._set(value=value)

    def getValue(self):
        """
        Gets the value of :py:attr:`value` or its default value.
        """
        return self.getOrDefault(self.value)

    def _transform(self, dataset):
        dataset_new = dataset.fillna(0)
        for i, j in zip(self.getInputCols(), self.getOutputCols()):
            dataset_new = dataset_new.withColumnRenamed(i, j)
        return dataset_new


###########
#构建测试数据
###########
import numpy as np

sentenceDataFrame = spark.createDataFrame([
    (0, 1, 2),
    (0, 1, 2),
    (1, 1, 2)
], ["label", "a", 'b'])
from pyspark.sql import functions

df = sentenceDataFrame.withColumn('c', functions.lit(np.nan))
df.show()

#############
#测试pipleline
#############
from pyspark.ml import Pipeline, PipelineModel, Transformer

blankTransformer = BlankTransformer(inputCols=["a", "b", "c"], outputCols=["a_1", "b_1", "c_1"])

p = Pipeline(stages=[blankTransformer])
# df = spark.sparkContext.parallelize([(1, None), (2, 1.0), (3, 0.5)]).toDF(["key", "value"])
pm = p.fit(df)
pm.transform(df).show()

###########################
#测试保存piplemodel,和加载测试
############################
pm.write().overwrite().save('./test/test.model')
pm2 = PipelineModel.load('./test/test.model')
print('matches?', pm2.stages[0].extractParamMap() == pm.stages[0].extractParamMap())
print(pm2.stages[0].extractParamMap())
pm2.transform(df).show()
