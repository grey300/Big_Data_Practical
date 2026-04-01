import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class MovieLens {

    // u.data: 0. user_id, 1. movie_id, 2. rating, 3. unix_timestamp
    /*
     * movie id | movie title | release date | video release date |
     * IMDb URL | unknown | Action | Adventure | Animation |
     * Children's | Comedy | Crime | Documentary | Drama | Fantasy |
     * Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
     * Thriller | War | Western |
     * The last 19 fields are the genres, a 1 indicates the movie
     * is of that genre, a 0 indicates it is not; movies can be in
     * several genres at once.
     * The movie ids are the ones used in the u.data data set.
     */
    public static class MovieMapper extends Mapper<Object, Text, Text, Text> {
        private final static IntWritable one = new IntWritable(1);

        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString();

            // Detect ratings file (u.data) - tab-separated
            if (line.contains("\t")) {
                String[] parts = line.split("\t");
                if (parts.length >= 3) {
                    String movieID = parts[1].trim();
                    String rating = parts[2].trim();
                    context.write(new Text(movieID), new Text("R|" + rating));
                }
            }
            // Detect movies file (u.item) - pipe-separated
            else if (line.contains("|")) {
                String[] parts = line.split("\\|");
                if (parts.length >= 2) {
                    String movieID = parts[0].trim();
                    String title = parts[1].trim();
                    context.write(new Text(movieID), new Text("M|" + title));
                }
            }

        }
    }

    // Reducer class that aggregates the word counts
    public static class MovieReducer extends Reducer<Text, Text, Text, DoubleWritable> {
        private DoubleWritable result = new DoubleWritable(); // Result object to store the aggregated count
        // Reduce function that processes the intermediate key-value pairs

        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            String movieTitle = "";
            List<Double> ratings = new ArrayList<>();

            for (Text val : values) {
                String s = val.toString();
                if (s.startsWith("M|")) {
                    movieTitle = s.substring(2); // extract title
                } else if (s.startsWith("R|")) {
                    ratings.add(Double.parseDouble(s.substring(2)));
                }
            }

            if (!movieTitle.isEmpty() && !ratings.isEmpty()) {
                double sum = 0;
                for (double r : ratings) {
                    sum += r;
                }
                double avg = sum / ratings.size();
                result.set(avg);
                context.write(new Text(movieTitle), result);
            }

        }
    }

    // Main method to configure and run the MapReduce job
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration(); // Create a new Hadoop configuration
        Job job = Job.getInstance(conf, "movie lens"); // Create a new job instance with a name
        job.setJarByClass(MovieLens.class); // Set the JAR file containing the job's code
        job.setMapperClass(MovieMapper.class); // Set the Mapper class
        // job.setNumReduceTasks(0);
        // job.setCombinerClass(MovieReducer.class); // Set the Combiner class
        // (optional but improves performance)
        job.setReducerClass(MovieReducer.class); // Set the Reducer class
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class); // Set the type of output keys
        job.setOutputValueClass(DoubleWritable.class); // Set the type of output values
        // Set input and output paths based on command-line arguments
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileInputFormat.addInputPath(job, new Path(args[1]));

        FileOutputFormat.setOutputPath(job, new Path(args[2]));

        // Submit the job and exit based on the job's success
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}