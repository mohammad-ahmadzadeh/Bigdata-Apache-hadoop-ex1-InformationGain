# Calculating Information Gain Using Apache Hadoop in Big Data Discussion
# Research Summary

In this study, the Apache Hadoop MapReduce parallel processing framework was utilized to compute the Information Gain criterion on large-scale data. The primary objective was to select the 100 most important features from among 2,381 features of the EMBER malware dataset. To achieve this, data comprising 5,000 malware and benign samples were first preprocessed and normalized, then loaded into the Hadoop system. By designing intelligent Mappers capable of locally processing data chunks and Reducers that aggregated the results, the process of calculating the final score for each feature was successfully accomplished.

## Evaluation

In the evaluation phase, 12 different experiments were designed and conducted with varying numbers of Mappers, Reducers, and chunk sizes. The results indicate that the best performance in terms of speed is achieved with a balanced configuration of **three Mappers, three Reducers, and a chunk size of 1,000 records**, which reduced execution time to approximately **80 seconds**. The accuracy of the model remained consistently stable across all configurations, at roughly **88 percent**. Furthermore, it was observed that the Map phase constitutes the **primary processing bottleneck** compared to the Reduce phase.

## Conclusion

Overall, this research demonstrates that intelligent parallelization can significantly accelerate the processing of large data volumes without compromising accuracy.
