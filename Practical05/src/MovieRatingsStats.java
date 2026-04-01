package src;

import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class MovieRatingsStats {

    public static class RatingsMapper extends Mapper<Object, Text, Text, IntWritable> {
        private Text movieId = new Text();
        private IntWritable ratingValue = new IntWritable();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty())
                return;

            String[] parts = line.split(",");
            if (parts.length != 3)
                return;

            movieId.set(parts[1].trim());
            ratingValue.set(Integer.parseInt(parts[2].trim()));
            context.write(movieId, ratingValue);
        }
    }

    public static class RatingsReducer extends Reducer<Text, IntWritable, Text, Text> {
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            int sum = 0;
            int count = 0;

            for (IntWritable v : values) {
                sum += v.get();
                count++;
            }

            double avg = (double) sum / count;
            context.write(key, new Text(String.format("%.1f,%d", avg, count)));
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: MovieRatingsStats <input> <output>");
            System.exit(2);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "movie ratings stats");
        job.setJarByClass(MovieRatingsStats.class);

        job.setMapperClass(RatingsMapper.class);
        job.setReducerClass(RatingsReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}