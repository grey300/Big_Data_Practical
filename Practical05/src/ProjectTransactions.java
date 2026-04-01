package src;

import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class ProjectTransactions {

    public static class ProjectMapper extends Mapper<Object, Text, Text, NullWritable> {
        private Text outValue = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty())
                return;

            String[] parts = line.split(",");
            if (parts.length != 4)
                return;

            String userId = parts[0].trim();
            String amount = parts[2].trim();

            outValue.set(userId + "," + amount);
            context.write(outValue, NullWritable.get());
        }
    }

    // reducer removes duplicates if same userID,amount appears multiple times
    public static class DedupReducer extends Reducer<Text, NullWritable, NullWritable, Text> {
        public void reduce(Text key, Iterable<NullWritable> values, Context context)
                throws IOException, InterruptedException {
            context.write(NullWritable.get(), key);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: ProjectTransactions <input> <output>");
            System.exit(2);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "project transactions");
        job.setJarByClass(ProjectTransactions.class);

        job.setMapperClass(ProjectMapper.class);
        job.setReducerClass(DedupReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(NullWritable.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}