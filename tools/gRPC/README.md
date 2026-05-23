# gRPC — High-Performance RPC Framework

gRPC is a modern, open-source, high-performance Remote Procedure Call (RPC) framework that uses Protocol Buffers for serialization and HTTP/2 for transport.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **.proto** | Service and message definitions in protobuf IDL |
| **Protobuf** | Binary serialization format (small, fast) |
| **Unary RPC** | Single request, single response |
| **Server Streaming** | Single request, stream of responses |
| **Client Streaming** | Stream of requests, single response |
| **Bidirectional Streaming** | Two independent streams of messages |
| **Interceptors** | Middleware for request/response interception |
| **Stub** | Client-side proxy generated from .proto |

## Protocol Buffer Definition

```protobuf
// model.proto
syntax = "proto3";

service ModelService {
  rpc Predict (PredictRequest) returns (PredictResponse);
  rpc PredictStream (PredictRequest) returns (stream PredictResponse);
}

message PredictRequest {
  string model_id = 1;
  repeated float features = 2 [packed = true];
}

message PredictResponse {
  float prediction = 1;
  float confidence = 2;
}
```

## Generate Stubs

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. model.proto
```

## Server Implementation

```python
from concurrent import futures
import grpc
import model_pb2, model_pb2_grpc

class ModelServicer(model_pb2_grpc.ModelServiceServicer):
    def Predict(self, request, context):
        result = run_model(request.model_id, request.features)
        return model_pb2.PredictResponse(
            prediction=result["score"],
            confidence=result["confidence"],
        )

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
model_pb2_grpc.add_ModelServiceServicer_to_server(ModelServicer(), server)
server.add_insecure_port("[::]:50051")
server.start()
server.wait_for_termination()
```

## Client

```python
import grpc
import model_pb2, model_pb2_grpc

with grpc.insecure_channel("localhost:50051") as channel:
    stub = model_pb2_grpc.ModelServiceStub(channel)
    response = stub.Predict(
        model_pb2.PredictRequest(
            model_id="v1",
            features=[1.2, 3.4, 5.6],
        )
    )
    print(response.prediction, response.confidence)
```

## Interceptors (Server-Side)

```python
class LoggingInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, handler_call_details):
        print(f"Method: {handler_call_details.method}")
        return continuation(handler_call_details)
```

## Integration Patterns

- **High-Throughput Model Serving**: Low-latency predictions at scale
- **Inter-Service Communication**: Microservices calling ML services
- **Streaming Inference**: Real-time predictions over data streams
- **Load Balancing**: gRPC's client-side load balancing for replicas

## References

- [gRPC Documentation](https://grpc.io/docs/)
- [gRPC Python](https://grpc.io/docs/languages/python/)
- [Protocol Buffers](https://protobuf.dev/)
