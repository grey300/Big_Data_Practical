package src;

import java.io.*;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class TopRatedMovies {

    private static final String DELIMITER = "\\t";

    // ---------------- JOB 1 ----------------
    // movieId -> avg,count

    public static class AvgCountMapper extends Mapper<LongWritable, Text, IntWritable, FloatWritable> {
        private final IntWritable movieId = new IntWritable();
        private final FloatWritable rating = new FloatWritable();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] fields = value.toString().trim().split(DELIMITER);
            if (fields.length >= 3) {
                try {
                    movieId.set(Integer.parseInt(fields[1]));
                    rating.set(Float.parseFloat(fields[2]));
                    context.write(movieId, rating);
                } catch (NumberFormatException e) {
                    // skip bad record
                }
            }
        }
    }

    public static class AvgCountReducer extends Reducer<IntWritable, FloatWritable, IntWritable, Text> {
        private final Text outValue = new Text();

        @Override
        public void reduce(IntWritable key, Iterable<FloatWritable> values, Context context)
                throws IOException, InterruptedException {

            float sum = 0;
            int count = 0;
            for (FloatWritable v : values) {
                sum += v.get();
                count++;
            }

            if (count > 0) {
                float avg = sum / count;
                outValue.set(avg + "," + count);
                context.write(key, outValue);
            }
        }
    }

    // ---------------- JOB 2 ----------------
    // keep count>=50, get title from u.item, output top 10

    public static class Top10Mapper extends Mapper<LongWritable, Text, NullWritable, Text> {
        private final Text outValue = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // input format: movieId \t avg,count
            String[] parts = value.toString().split("\\t");
            if (parts.length != 2)
                return;

            try {
                int movieId = Integer.parseInt(parts[0]);
                String[] stats = parts[1].split(",");
                float avg = Float.parseFloat(stats[0]);
                int count = Integer.parseInt(stats[1]);

                if (count >= 50) {
                    outValue.set(movieId + "," + avg);
                    context.write(NullWritable.get(), outValue);
                }
            } catch (Exception e) {
                // skip bad row
            }
        }
    }

    public static class Top10Reducer extends Reducer<NullWritable, Text, Text, FloatWritable> {
        private final Map<Integer, String> movieTitles = new HashMap<>();

        @Override
        protected void setup(Context context) throws IOException {
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles == null)
                return;

            for (URI uri : cacheFiles) {
                Path path = new Path(uri.getPath());
                String fileName = path.getName();

                if (fileName.equals("u.item")) {
                    BufferedReader br = new BufferedReader(new FileReader(fileName));
                    String line;
                    while ((line = br.readLine()) != null) {
                        String[] fields = line.split("\\|");
                        if (fields.length >= 2) {
                            try {
                                int movieId = Integer.parseInt(fields[0]);
                                String title = fields[1];
                                movieTitles.put(movieId, title);
                            } catch (NumberFormatException e) {
                                // skip
                            }
                        }
                    }
                    br.close();
                }
            }
        }

        @Override
        public void reduce(NullWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            List<MovieRecord> list = new ArrayList<>();

            for (Text val : values) {
                String[] parts = val.toString().split(",");
                if (parts.length != 2)
                    continue;

                try {
                    int movieId = Integer.parseInt(parts[0]);
                    float avg = Float.parseFloat(parts[1]);
                    list.add(new MovieRecord(movieId, avg));
                } catch (NumberFormatException e) {
                    // skip
                }
            }

            list.sort((a, b) -> Float.compare(b.avg, a.avg));

            int limit = Math.min(10, list.size());
            for (int i = 0; i < limit; i++) {
                MovieRecord r = list.get(i);
                String title = movieTitles.getOrDefault(r.movieId, "MovieID_" + r.movieId);
                context.write(new Text(title), new FloatWritable(r.avg));
            }
        }
    }

    static class MovieRecord {
        int movieId;
        float avg;

        MovieRecord(int movieId, float avg) {
            this.movieId = movieId;
            this.avg = avg;
        }
    }

    public static void main(String[] args) throws Exception {
        // args[0] = u.data input
        // args[1] = temp output
        // args[2] = final output
        // args[3] = u.item path (local/container path for cache)
        if (args.length != 4) {
            System.err.println("Usage: TopRatedMovies <u.data input> <temp output> <final output> <u.item cache file>");
            System.exit(2);
        }

        Configuration conf1 = new Configuration();
        Job job1 = Job.getInstance(conf1, "task b - avg and count per movie");

        job1.setJarByClass(TopRatedMovies.class);
        job1.setMapperClass(AvgCountMapper.class);
        job1.setReducerClass(AvgCountReducer.class);

        job1.setMapOutputKeyClass(IntWritable.class);
        job1.setMapOutputValueClass(FloatWritable.class);
        job1.setOutputKeyClass(IntWritable.class);
        job1.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job1, new Path(args[0]));
        FileOutputFormat.setOutputPath(job1, new Path(args[1]));

        if (!job1.waitForCompletion(true)) {
            System.exit(1);
        }

        Configuration conf2 = new Configuration();
        Job job2 = Job.getInstance(conf2, "task b - top 10 highest rated movies");

        job2.setJarByClass(TopRatedMovies.class);
        job2.setMapperClass(Top10Mapper.class);
        job2.setReducerClass(Top10Reducer.class);
        job2.setNumReduceTasks(1);

        job2.setMapOutputKeyClass(NullWritable.class);
        job2.setMapOutputValueClass(Text.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(FloatWritable.class);

        job2.addCacheFile(new URI(args[3] + "#u.item"));

        FileInputFormat.addInputPath(job2, new Path(args[1]));
        FileOutputFormat.setOutputPath(job2, new Path(args[2]));

        System.exit(job2.waitForCompletion(true) ? 0 : 1);
    }
}