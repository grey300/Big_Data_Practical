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

public class PreferredGenreByOccupation {

    private static final String DELIMITER = "\\t";

    private static final String[] GENRES = {
            "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    };

    // -------- JOB 1 --------
    // Join ratings with users -> occupation,movieId

    public static class UserRatingMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        private final IntWritable userIdKey = new IntWritable();
        private final Text outValue = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            String[] fields = line.split(DELIMITER);

            if (fields.length == 5) {
                // u.user
                try {
                    int userId = Integer.parseInt(fields[0]);
                    String occupation = fields[3];
                    userIdKey.set(userId);
                    outValue.set("U|" + occupation);
                    context.write(userIdKey, outValue);
                } catch (Exception e) {
                    // skip
                }
            } else if (fields.length >= 4) {
                // u.data
                try {
                    int userId = Integer.parseInt(fields[0]);
                    String movieId = fields[1];
                    userIdKey.set(userId);
                    outValue.set("R|" + movieId);
                    context.write(userIdKey, outValue);
                } catch (Exception e) {
                    // skip
                }
            }
        }
    }

    public static class UserRatingReducer extends Reducer<IntWritable, Text, Text, IntWritable> {
        private final Text outKey = new Text();
        private final IntWritable outValue = new IntWritable();

        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            String occupation = null;
            List<Integer> movieIds = new ArrayList<>();

            for (Text val : values) {
                String s = val.toString();
                if (s.startsWith("U|")) {
                    occupation = s.substring(2);
                } else if (s.startsWith("R|")) {
                    try {
                        movieIds.add(Integer.parseInt(s.substring(2)));
                    } catch (Exception e) {
                        // skip
                    }
                }
            }

            if (occupation != null) {
                outKey.set(occupation);
                for (Integer movieId : movieIds) {
                    outValue.set(movieId);
                    context.write(outKey, outValue);
                }
            }
        }
    }

    // -------- JOB 2 --------
    // occupation,movieId -> occupation,genre counts
    public static class GenreCountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        private final Map<Integer, List<String>> movieGenres = new HashMap<>();
        private final Text outKey = new Text();
        private final static IntWritable ONE = new IntWritable(1);

        @Override
        protected void setup(Context context) throws IOException {
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles == null)
                return;

            for (URI uri : cacheFiles) {
                Path path = new Path(uri.getPath());
                if (path.getName().equals("u.item")) {
                    BufferedReader br = new BufferedReader(new FileReader("u.item"));
                    String line;
                    while ((line = br.readLine()) != null) {
                        String[] fields = line.split("\\|");
                        if (fields.length >= 24) {
                            try {
                                int movieId = Integer.parseInt(fields[0]);
                                List<String> genres = new ArrayList<>();

                                for (int i = 0; i < 19; i++) {
                                    int idx = 5 + i;
                                    if ("1".equals(fields[idx])) {
                                        genres.add(GENRES[i]);
                                    }
                                }
                                movieGenres.put(movieId, genres);
                            } catch (Exception e) {
                                // skip
                            }
                        }
                    }
                    br.close();
                }
            }
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // input: occupation \t movieId
            String[] parts = value.toString().split("\\t");
            if (parts.length != 2)
                return;

            String occupation = parts[0];

            try {
                int movieId = Integer.parseInt(parts[1]);
                List<String> genres = movieGenres.get(movieId);
                if (genres != null) {
                    for (String genre : genres) {
                        outKey.set(occupation + "|" + genre);
                        context.write(outKey, ONE);
                    }
                }
            } catch (Exception e) {
                // skip
            }
        }
    }

    public static class GenreCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private final IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            int sum = 0;
            for (IntWritable v : values) {
                sum += v.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    // -------- JOB 3 --------
    // find max genre per occupation
    public static class FinalMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text outKey = new Text();
        private final Text outValue = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // input: occupation|genre \t count
            String[] parts = value.toString().split("\\t");
            if (parts.length != 2)
                return;

            String[] left = parts[0].split("\\|");
            if (left.length != 2)
                return;

            String occupation = left[0];
            String genre = left[1];
            String count = parts[1];

            outKey.set(occupation);
            outValue.set(genre + "|" + count);
            context.write(outKey, outValue);
        }
    }

    public static class FinalReducer extends Reducer<Text, Text, Text, Text> {
        private final Text outValue = new Text();

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            String bestGenre = "";
            int maxCount = -1;

            for (Text val : values) {
                String[] parts = val.toString().split("\\|");
                if (parts.length != 2)
                    continue;

                String genre = parts[0];
                int count = Integer.parseInt(parts[1]);

                if (count > maxCount) {
                    maxCount = count;
                    bestGenre = genre;
                }
            }

            outValue.set(bestGenre);
            context.write(key, outValue);
        }
    }

    public static void main(String[] args) throws Exception {
        // args[0] = u.data
        // args[1] = u.user
        // args[2] = temp1
        // args[3] = temp2
        // args[4] = final out
        // args[5] = u.item cache file
        if (args.length != 6) {
            System.err.println(
                    "Usage: PreferredGenreByOccupation <u.data> <u.user> <temp1> <temp2> <out> <u.item cache>");
            System.exit(2);
        }

        Configuration conf1 = new Configuration();
        Job job1 = Job.getInstance(conf1, "task d - join users and ratings");

        job1.setJarByClass(PreferredGenreByOccupation.class);
        job1.setMapperClass(UserRatingMapper.class);
        job1.setReducerClass(UserRatingReducer.class);

        job1.setMapOutputKeyClass(IntWritable.class);
        job1.setMapOutputValueClass(Text.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job1, new Path(args[0]));
        FileInputFormat.addInputPath(job1, new Path(args[1]));
        FileOutputFormat.setOutputPath(job1, new Path(args[2]));

        if (!job1.waitForCompletion(true))
            System.exit(1);

        Configuration conf2 = new Configuration();
        Job job2 = Job.getInstance(conf2, "task d - count genres by occupation");

        job2.setJarByClass(PreferredGenreByOccupation.class);
        job2.setMapperClass(GenreCountMapper.class);
        job2.setReducerClass(GenreCountReducer.class);

        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(IntWritable.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(IntWritable.class);

        job2.addCacheFile(new URI(args[5] + "#u.item"));

        FileInputFormat.addInputPath(job2, new Path(args[2]));
        FileOutputFormat.setOutputPath(job2, new Path(args[3]));

        if (!job2.waitForCompletion(true))
            System.exit(1);

        Configuration conf3 = new Configuration();
        Job job3 = Job.getInstance(conf3, "task d - preferred genre per occupation");

        job3.setJarByClass(PreferredGenreByOccupation.class);
        job3.setMapperClass(FinalMapper.class);
        job3.setReducerClass(FinalReducer.class);

        job3.setMapOutputKeyClass(Text.class);
        job3.setMapOutputValueClass(Text.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job3, new Path(args[3]));
        FileOutputFormat.setOutputPath(job3, new Path(args[4]));

        System.exit(job3.waitForCompletion(true) ? 0 : 1);
    }
}