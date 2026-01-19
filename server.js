const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

app.use(express.static(path.join(__dirname, 'public')));

// DATABASE (In-Memory)
// Structure: { socketId: { name, roll, logs: [], ... } }
let activeStudents = {};

io.on('connection', (socket) => {
    // 1. Send current data to new admins
    socket.emit('update-dashboard', Object.values(activeStudents));

    // 2. Student Connects
    socket.on('student-connect', (data) => {
        activeStudents[socket.id] = {
            id: socket.id,
            name: data.name,
            roll: data.roll,
            sap: data.sap,
            startTime: new Date().toLocaleTimeString(),
            endTime: '-',
            riskScore: 'Normal',
            logs: [] // <-- NEW: Array to store violation history
        };
        io.emit('update-dashboard', Object.values(activeStudents));
    });

    // 3. Status Update (The Logging Engine)
    socket.on('student-status-update', (statusText) => {
        if (activeStudents[socket.id]) {
            const student = activeStudents[socket.id];
            student.riskScore = statusText;

            // Log if it's a violation
            if (statusText.includes("VIOLATION")) {
                const logEntry = {
                    time: new Date().toLocaleTimeString(),
                    violation: statusText
                };
                student.logs.push(logEntry);
            }

            io.emit('update-dashboard', Object.values(activeStudents));
        }
    });

    // 4. Handle Disconnect
    socket.on('disconnect', () => {
        if (activeStudents[socket.id]) {
            activeStudents[socket.id].endTime = new Date().toLocaleTimeString();
            activeStudents[socket.id].riskScore = 'Offline';
            io.emit('update-dashboard', Object.values(activeStudents));
        }
    });
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});