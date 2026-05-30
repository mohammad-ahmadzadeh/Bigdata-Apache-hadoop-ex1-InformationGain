// AnalyticsFeatureRanker.java
//Mohammad Agmadzadeh - IAU university, srbiau branch

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.AbstractMap.SimpleEntry;


public class AnalyticsFeatureRanker {
    
    public static void main(String[] arguments) throws Exception {
        if (arguments.length < 2) {
            System.err.println("Usage: hadoop jar analytics.jar AnalyticsFeatureRanker <input path> <output path>");
            System.exit(1);
        }
        
        String dataInputPath = arguments[0];
        String resultOutputPath = arguments[1];
        
        Configuration jobConfig = new Configuration();
        
        int targetColumnIndex = findTargetColumn(jobConfig, dataInputPath);
        System.out.println("Target column index: " + targetColumnIndex);
        jobConfig.setInt("target.column.index", targetColumnIndex);
        
        Map<String, Integer> classDistribution = calculateClassDistribution(jobConfig, dataInputPath, targetColumnIndex);
        
        String distributionString = buildDistributionString(classDistribution);
        int totalRows = classDistribution.values().stream().mapToInt(Integer::intValue).sum();
        
        printDatasetStats(totalRows, distributionString, classDistribution.size());
        
        if (totalRows == 0) {
            System.err.println("ERROR: No valid records found. Check CSV format.");
            System.exit(1);
        }
        
        jobConfig.set("global.class.distribution", distributionString);
        jobConfig.setInt("total.row.count", totalRows);
        
        FileSystem hdfsClient = FileSystem.get(jobConfig);
        Path outputFolder = new Path(resultOutputPath);
        if (hdfsClient.exists(outputFolder)) {
            hdfsClient.delete(outputFolder, true);
            System.out.println("Removed existing output: " + resultOutputPath);
        }
        
        Job mapReduceJob = Job.getInstance(jobConfig, "Information Gain Feature Selection");
        mapReduceJob.setJarByClass(AnalyticsFeatureRanker.class);
        
        mapReduceJob.setMapperClass(FeatureExtractMapper.class);
        mapReduceJob.setCombinerClass(PartialAggregateCombiner.class);
        mapReduceJob.setReducerClass(GainComputeReducer.class);
        
        mapReduceJob.setMapOutputKeyClass(Text.class);
        mapReduceJob.setMapOutputValueClass(Text.class);
        mapReduceJob.setOutputKeyClass(Text.class);
        mapReduceJob.setOutputValueClass(DoubleWritable.class);
        
        mapReduceJob.setInputFormatClass(TextInputFormat.class);
        mapReduceJob.setOutputFormatClass(TextOutputFormat.class);
        
        mapReduceJob.setNumReduceTasks(10);
        
        FileInputFormat.addInputPath(mapReduceJob, new Path(dataInputPath));
        FileOutputFormat.setOutputPath(mapReduceJob, new Path(resultOutputPath));
        
        printJobInfo(dataInputPath, resultOutputPath);
        
        boolean jobSuccess = mapReduceJob.waitForCompletion(true);
        
        if (jobSuccess) {
            System.out.println("\nJob completed successfully!");
            showTopRankedFeatures(jobConfig, resultOutputPath);
        } else {
            System.err.println("\nJob failed!");
            System.exit(1);
        }
        
        System.exit(jobSuccess ? 0 : 1);
    }
    
    private static String buildDistributionString(Map<String, Integer> distribution) {
        return distribution.entrySet().stream()
            .map(entry -> entry.getKey() + ":" + entry.getValue())
            .collect(Collectors.joining(","));
    }
    
    private static void printDatasetStats(int totalRows, String distribution, int uniqueClasses) {
        System.out.println("=== Dataset Statistics ===");
        System.out.println("Total records: " + totalRows);
        System.out.println("Class distribution: " + distribution);
        System.out.println("Unique classes: " + uniqueClasses);
    }
    
    private static void printJobInfo(String inputPath, String outputPath) {
        System.out.println("\n=== Starting Hadoop Job ===");
        System.out.println("Input: " + inputPath);
        System.out.println("Output: " + outputPath);
        System.out.println("===========================\n");
    }
    
    private static int findTargetColumn(Configuration config, String filePath) throws Exception {
        FileSystem hdfs = FileSystem.get(config);
        Path targetFile = new Path(filePath);
        
        if (!hdfs.exists(targetFile)) {
            System.err.println("File not found: " + filePath);
            return -1;
        }
        
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(hdfs.open(targetFile)))) {
            String headerLine = reader.readLine();
            if (headerLine == null) return -1;
            
            String[] columnNames = headerLine.split(",");
            System.out.println("Header has " + columnNames.length + " columns");
            
            for (int i = 0; i < columnNames.length; i++) {
                String name = columnNames[i].trim().toLowerCase();
                if (name.equals("label") || name.equals("class") || 
                    name.equals("avclass") || name.equals("target")) {
                    System.out.println("Target column at index " + i + ": '" + columnNames[i] + "'");
                    return i;
                }
            }
            
            int lastIndex = columnNames.length - 1;
            System.out.println("Using last column as target (index " + lastIndex + "): '" + 
                               columnNames[lastIndex] + "'");
            return lastIndex;
        }
    }
    
    private static Map<String, Integer> calculateClassDistribution(Configuration config, String filePath, int targetIndex) 
            throws Exception {
        
        Map<String, Integer> counter = new HashMap<>();
        FileSystem hdfs = FileSystem.get(config);
        Path targetFile = new Path(filePath);
        
        if (!hdfs.exists(targetFile)) return counter;
        
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(hdfs.open(targetFile)))) {
            boolean isFirstLine = true;
            String line;
            
            while ((line = reader.readLine()) != null) {
                if (isFirstLine) {
                    isFirstLine = false;
                    continue;
                }
                
                String[] parts = line.split(",");
                if (targetIndex >= 0 && targetIndex < parts.length) {
                    String classValue = parts[targetIndex].trim();
                    if (!classValue.isEmpty() && !classValue.equalsIgnoreCase("label")) {
                        counter.put(classValue, counter.getOrDefault(classValue, 0) + 1);
                    }
                }
            }
        }
        
        return counter;
    }
    
    private static void showTopRankedFeatures(Configuration config, String outputFolder) throws Exception {
        FileSystem hdfs = FileSystem.get(config);
        Path resultPath = new Path(outputFolder);
        
        if (!hdfs.exists(resultPath)) return;
        
        List<SimpleEntry<String, Double>> rankedFeatures = new ArrayList<>();
        
        org.apache.hadoop.fs.FileStatus[] files = hdfs.listStatus(resultPath);
        for (org.apache.hadoop.fs.FileStatus fileInfo : files) {
            if (fileInfo.getPath().getName().startsWith("part-")) {
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(hdfs.open(fileInfo.getPath())))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] segments = line.split("\t");
                        if (segments.length == 2) {
                            try {
                                rankedFeatures.add(new SimpleEntry<>(segments[0], Double.parseDouble(segments[1])));
                            } catch (NumberFormatException ignored) {}
                        }
                    }
                }
            }
        }
        
        rankedFeatures.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        
        System.out.println("\n" + "=".repeat(80));
        System.out.println("TOP 100 FEATURES BY INFORMATION GAIN");
        System.out.println("=".repeat(80));
        System.out.printf("%-5s %-50s %-15s%n", "Rank", "Feature Name", "Score");
        System.out.println("-".repeat(80));
        
        for (int i = 0; i < Math.min(100, rankedFeatures.size()); i++) {
            SimpleEntry<String, Double> entry = rankedFeatures.get(i);
            String displayName = entry.getKey();
            if (displayName.length() > 50) {
                displayName = displayName.substring(0, 47) + "...";
            }
            System.out.printf("%-5d %-50s %-15.6f%n", i + 1, displayName, entry.getValue());
        }
    }
    
    // ==================== MAPPER ====================
    public static class FeatureExtractMapper extends Mapper<LongWritable, Text, Text, Text> {
        
        private boolean headerProcessed = false;
        private int targetIdx = -1;
        private List<String> featureNames = new ArrayList<>();
        
        @Override
        protected void setup(Context ctx) {
            targetIdx = ctx.getConfiguration().getInt("target.column.index", -1);
        }
        
        @Override
        protected void map(LongWritable rowId, Text rowContent, Context ctx)
                throws IOException, InterruptedException {
            
            String line = rowContent.toString();
            if (line.trim().isEmpty()) return;
            
            String[] values = line.split(",");
            
            if (!headerProcessed) {
                processHeader(values, ctx);
                headerProcessed = true;
                return;
            }
            
            if (targetIdx >= 0 && targetIdx < values.length) {
                String targetVal = values[targetIdx].trim();
                
                if (!targetVal.isEmpty() && !targetVal.equalsIgnoreCase("label")) {
                    emitFeatureValuePairs(values, targetVal, ctx);
                }
            }
        }
        
        private void processHeader(String[] headers, Context ctx) {
            for (int i = 0; i < headers.length; i++) {
                featureNames.add(headers[i].trim());
            }
            
            if (targetIdx == -1 && headers.length > 0) {
                targetIdx = headers.length - 1;
                ctx.getConfiguration().setInt("target.column.index", targetIdx);
            }
        }
        
        private void emitFeatureValuePairs(String[] values, String targetValue, Context ctx) 
                throws IOException, InterruptedException {
            
            for (int i = 0; i < values.length; i++) {
                if (i != targetIdx && i < featureNames.size()) {
                    String featureName = featureNames.get(i);
                    String featureVal = values[i].trim();
                    
                    if (!featureVal.isEmpty()) {
                        ctx.write(new Text(featureName), new Text(featureVal + "|" + targetValue));
                    }
                }
            }
        }
    }
    
    // ==================== COMBINER ====================
    public static class PartialAggregateCombiner extends Reducer<Text, Text, Text, Text> {
        
        @Override
        protected void reduce(Text featureKey, Iterable<Text> incomingValues, Context ctx)
                throws IOException, InterruptedException {
            
            Map<String, Map<String, Integer>> valueClassCounts = new HashMap<>();
            
            for (Text item : incomingValues) {
                String[] parts = item.toString().split("\\|");
                if (parts.length == 2) {
                    String featureVal = parts[0];
                    String classVal = parts[1];
                    
                    valueClassCounts.putIfAbsent(featureVal, new HashMap<>());
                    Map<String, Integer> classCounter = valueClassCounts.get(featureVal);
                    classCounter.put(classVal, classCounter.getOrDefault(classVal, 0) + 1);
                }
            }
            
            for (Map.Entry<String, Map<String, Integer>> entry : valueClassCounts.entrySet()) {
                String featureVal = entry.getKey();
                Map<String, Integer> counts = entry.getValue();
                
                String combined = counts.entrySet().stream()
                    .map(e -> e.getKey() + ":" + e.getValue())
                    .collect(Collectors.joining(","));
                
                ctx.write(featureKey, new Text(featureVal + "|" + combined));
            }
        }
    }
    
    // ==================== REDUCER ====================
    public static class GainComputeReducer extends Reducer<Text, Text, Text, DoubleWritable> {
        
        private int totalRows = 0;
        private Map<String, Integer> globalClassCounts = new HashMap<>();
        
        @Override
        protected void setup(Context ctx) {
            totalRows = ctx.getConfiguration().getInt("total.row.count", 0);
            String distribution = ctx.getConfiguration().get("global.class.distribution", "");
            
            if (!distribution.isEmpty()) {
                String[] pairs = distribution.split(",");
                for (String pair : pairs) {
                    String[] kv = pair.split(":");
                    if (kv.length == 2) {
                        globalClassCounts.put(kv[0], Integer.parseInt(kv[1]));
                    }
                }
            }
        }
        
        @Override
        protected void reduce(Text featureName, Iterable<Text> aggregatedData, Context ctx)
                throws IOException, InterruptedException {
            
            Map<String, Map<String, Integer>> valueClassMatrix = new HashMap<>();
            
            for (Text item : aggregatedData) {
                String[] segments = item.toString().split("\\|");
                if (segments.length == 2) {
                    String featureVal = segments[0];
                    String classData = segments[1];
                    
                    valueClassMatrix.putIfAbsent(featureVal, new HashMap<>());
                    Map<String, Integer> classCounter = valueClassMatrix.get(featureVal);
                    
                    String[] classEntries = classData.split(",");
                    for (String entry : classEntries) {
                        String[] kv = entry.split(":");
                        if (kv.length == 2) {
                            classCounter.put(kv[0], classCounter.getOrDefault(kv[0], 0) + Integer.parseInt(kv[1]));
                        }
                    }
                }
            }
            
            double informationGain = computeInfoGain(valueClassMatrix);
            ctx.write(featureName, new DoubleWritable(informationGain));
        }
        
        private double computeInfoGain(Map<String, Map<String, Integer>> valueClassMatrix) {
            if (totalRows == 0) return 0.0;
            
            double totalEntropy = calculateEntropy(globalClassCounts, totalRows);
            double weightedConditionalEntropy = 0.0;
            
            for (Map<String, Integer> classCounts : valueClassMatrix.values()) {
                int subsetSize = classCounts.values().stream().mapToInt(Integer::intValue).sum();
                double weight = (double) subsetSize / totalRows;
                double entropy = calculateEntropy(classCounts, subsetSize);
                weightedConditionalEntropy += weight * entropy;
            }
            
            return totalEntropy - weightedConditionalEntropy;
        }
        
        private double calculateEntropy(Map<String, Integer> classCounts, int total) {
            if (total == 0) return 0.0;
            
            return -classCounts.values().stream()
                .filter(count -> count > 0)
                .mapToDouble(count -> {
                    double prob = (double) count / total;
                    return prob * (Math.log(prob) / Math.log(2));
                })
                .sum();
        }
    }
}