package src;

import java.io.IOException;

import javax.naming.Context;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.io.Text;

public class AvgRatingByGender {

    private static final String DELIMITER = "\\t";

    // -------- JOB 1: join user and rating on userId --------

    public static class JoinMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        private final IntWritable userIdKey = new IntWritable();
        private final Text outValue = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();

            String[] fields;

            if (line.contains("|")) {
                // u.user
                fields = line.split("\\|");
            } else {
                // u.data
                fields = line.split("\\t");
            }

            // u.user -> userId|age|gender|occupation|zip
            if (fields.length == 5) {
                try {
                    int userId = Integer.parseInt(fields[0]);
                    String gender = fields[2];
                    userIdKey.set(userId);
                    outValue.set("U|" + gender);
                    context.write(userIdKey, outValue);
                } catch (Exception e) {
                    // skip
                }
            }
            // u.data -> userId|movieId|rating|timestamp
            else if (fields.length >= 4) {
                try {
                    int userId = Integer.parseInt(fields[0]);
                    String rating = fields[2];
                    userIdKey.set(userId);
                    outValue.set("R|" + rating);
                    context.write(userIdKey, outValue);
                } catch (Exception e) {
                    // skip
                }
            }
        }
    }

    public static class JoinReducer extends Reducer<IntWritable, Text, Text, FloatWritable> {
        private final Text outKey = new Text();
        private final FloatWritable outValue = new FloatWritable();

        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            String gender = null;
            java.util.List<Float> ratings = new java.util.ArrayList<>();

            for (Text val : values) {
                String s = val.toString();
                if (s.startsWith("U|")) {
                    gender = s.substring(2);
                } else if (s.startsWith("R|")) {
                    try {
                        ratings.add(Float.parseFloat(s.substring(2)));
                    } catch (Exception e) {
                        // skip
                    }
                }
            }

            if (gender != null) {
                outKey.set(gender);
                for (Float r : ratings) {
                    outValue.set(r);
                    context.write(outKey, outValue);
                }
            }
        }
    }

    // -------- JOB 2: average by gender --------

    public static class GenderAvgReducer extends Reducer<Text, FloatWritable, Text, FloatWritable> {
        private final FloatWritable result = new FloatWritable();

        @Override
        public void reduce(Text key, Iterable<FloatWritable> values, Context context)
                throws IOException, InterruptedException {

            float sum = 0;
            int count = 0;
            for (FloatWritable v : values) {
                sum += v.get();
                count++;
            }

            if (count > 0) {
                result.set(sum / count);
                context.write(key, result);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        // args[0] = ratings input path
        // args[1] = users input path
        // args[2] = temp output
        // args[3] = final output
        if (args.length != 4) {
            System.err.println("Usage: AvgRatingByGender <u.data> <u.user> <temp out> <final out>");
            System.exit(2);
        }

        Configuration conf1 = new Configuration();
        Job job1 = Job.getInstance(conf1, "task c - join ratings and users");

        job1.setJarByClass(AvgRatingByGender.class);
        job1.setMapperClass(JoinMapper.class);
        job1.setReducerClass(JoinReducer.class);

        job1.setMapOutputKeyClass(IntWritable.class);
        job1.setMapOutputValueClass(Text.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(FloatWritable.class);

        FileInputFormat.addInputPath(job1, new Path(args[0]));
        FileInputFormat.addInputPath(job1, new Path(args[1]));
        FileOutputFormat.setOutputPath(job1, new Path(args[2]));

        if (!job1.waitForCompletion(true)) {
            System.exit(1);
        }

        Configuration conf2 = new Configuration();
        Job job2 = Job.getInstance(conf2, "task c - average rating by gender");

        job2.setJarByClass(AvgRatingByGender.class);
        job2.setMapperClass(Mapper.class);
        job2.setReducerClass(GenderAvgReducer.class);

        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(FloatWritable.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(FloatWritable.class);

        FileInputFormat.addInputPath(job2, new Path(args[2]));
        FileOutputFormat.setOutputPath(job2, new Path(args[3]));

        System.exit(job2.waitForCompletion(true) ? 0 : 1);
    }
}