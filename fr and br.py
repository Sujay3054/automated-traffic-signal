import React, { useState, useEffect } from "react";

function App() {
  const [frame, setFrame] = useState(null);
  const [vehicleCount, setVehicleCount] = useState(0);
  const [signalStatus, setSignalStatus] = useState("RED");

  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8000/ws");

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setVehicleCount(data.vehicle_count);
      setSignalStatus(data.signal_status);
      setFrame(`data:image/jpeg;base64,${data.frame}`);
    };

    return () => socket.close();
  }, []);

  return (
    <div className="container">
      <h1>Traffic Signal Control</h1>
      <img src={frame} alt="Live Traffic" />
      <h2>Vehicle Count: {vehicleCount}</h2>
      <h2 style={{ color: signalStatus === "GREEN" ? "green" : "red" }}>
        Signal: {signalStatus}
      </h2>
    </div>
  );
}

export default App;
