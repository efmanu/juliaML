#Example 2: Image Classification with a Convolutional Neural Network (CNN)
using Flux
using MLDatasets: MNIST
using Flux: onehotbatch, onecold, throttle
using Statistics

# Load the MNIST dataset
train_data = MNIST(Float32, :train)
train_X = Flux.unsqueeze(train_data.features,3)
train_y = Flux.onehotbatch(train_data.targets, 0:9)

test_data = MNIST(Float32, :test)
test_X = Flux.unsqueeze(test_data.features,3)
test_y = Flux.onehotbatch(test_data.targets, 0:9)


data = Flux.DataLoader(
    (train_X, train_y);
    batchsize=10,
    shuffle=true
)



# Define a simple CNN model
model = Chain(
    Conv((2, 2), 1=>8, relu),
    x -> maxpool(x, (2,2)),
    Flux.flatten,
    Dense(13*13*8, 10),
    softmax
)

# Define loss function and optimizer
loss(x, y) = Flux.crossentropy(model(x), y)
optimizer = Flux.Descent()

# Training the model
for epoch in 1:10
    @info "Epoch $epoch and loss $(loss(train_X,train_y))"
    Flux.train!(loss, Flux.params(model), data, optimizer)
end

# Testing the model
accuracy = mean(onecold(model(test_X)) .== onecold(test_y))
println("Test accuracy: $accuracy")