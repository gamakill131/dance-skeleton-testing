import express from 'express'
import * as http from 'http'
import { LevelController } from './controllers/level_controller.js';
import * as dotenv from 'dotenv'
import connectDB from './config/db.js';
import mongoose from 'mongoose';


dotenv.config();

const app = express();
const server = http.createServer(app);
const port = process.env.PORT || 5000;

connectDB();

app.use(express.json());
app.use('/levels', LevelController);

mongoose.connection.once('open', () => {
    server.listen(port, () => {
        console.log("Server started on port " + port);
    })
});