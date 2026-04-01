package src;

// Import necessary Hadoop libraries
import java.io.IOException;

import javax.naming.Context;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

// Main class for the Word Count program
public class WordCount {

    // Mapper class that processes input text
    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1); // Value to represent each word occurrence
        private org.w3c.dom.Text word = new Text(); // Text object to store each word
        // Map function that processes each line of input

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // Split the input line into words based on whitespace
            String[] words = value.toString().split("\\s+");
            for (String w : words) {
                word.set(w); // Set the current word
                IntWritable length = new IntWritable(w.length());
                context.write(word, length); // Write the word and its count (1) to the context
            }
        }
    }

    // Reducer class that aggregates the word counts
    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable(); // Result object to store the aggregated count
        // Reduce function that processes the intermediate key-value pairs

        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0; // Initialize sum to 0
            for (IntWritable val : values) {
                sum += val.get(); // Add up the counts for the current word
            }
            result.set(sum); // Set the total count for the current word
            context.write(key, result); // Write the word and its total count to the context
        }
    }

    // Main method to configure and run the MapReduce job
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration(); // Create a new Hadoop configuration
        Job job = Job.getInstance(conf, "word count"); // Create a new job instance with a name

        job.setJarByClass(WordCount.class); // Set the JAR file containing the job's code
        job.setMapperClass(TokenizerMapper.class); // Set the Mapper class
        job.setNumReduceTasks(0);
        // job.setCombinerClass(IntSumReducer.class); // Set the Combiner class
        // (optional but improves performance)
        // job.setReducerClass(IntSumReducer.class); // Set the Reducer class

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        // job.setOutputKeyClass(Text.class); // Set the type of output keys
        // job.setOutputValueClass(IntWritable.class); // Set the type of output values

        // Set input and output paths based on command-line arguments
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        // Submit the job and exit based on the job's success
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
