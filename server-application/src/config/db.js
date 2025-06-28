import mongoose from 'mongoose';
const connectDB = async () => {
    try {
        if (!process.env.MONGO_URI) {
            throw new Error("Mongo URI not set in env file");
        }
        const conn = await mongoose.connect(process.env.MONGO_URI);
        console.log(`Connected to database: ${conn.connection.host}`);
    }
    catch (error) {
        console.log(error);
        process.exit(1);
    }
};
export default connectDB;
