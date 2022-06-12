def model_DG1(train_x,train_y,test_x,test_y):

    # Data preprocessing
    [_,m, n] = train_x.shape
    # 建立模型
    inputs = Input(shape=(m,n))
    layer0 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    layer1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(layer0)
    temp = tf.concat([layer0, layer1], axis=1)
    layer2 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(temp)
    layer3 = tf.concat([layer0,layer1,layer2], axis=1)
    layer7 = GRU(64, return_sequences=True)(layer3)
    layer7 = Dropout(0.7)(layer7)
    layer7 = layer7+layer3
    layer9 = Flatten()(layer7)
    layer10 = Dropout(0.5)(layer9)
    layer11 = Dense(64, activation='relu',kernel_regularizer=regularizers.l1(0.0001))(layer10)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid",kernel_regularizer=regularizers.l1(0.0001))(layer11)

    model = tf.keras.Model(inputs, outputs)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=0.001),
                  metrics=['acc'])
    model.fit(train_x, train_y, epochs=10, batch_size=100,validation_data=(test_x,test_y))
    x = model.predict(test_x)
    x1 = 1-x
    x = np.hstack((x1,x))
    return x

def model_Dense_GRU(train_x,train_y,test_x,test_y):
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
    callbacks_list = [lr_scheduler, ]

    # data preprocessing
    [m, n] = train_x.shape
    train_x = np.reshape(train_x, (-1, 1, n))
    test_x = np.reshape(test_x, (-1, 1, n))

    # 转换独热编码形式
    train_y = utils.to_categorical(train_y)
    test_y = utils.to_categorical(test_y)

    # 建立模型
    inputs = Input(shape=(1, n))
    layer0 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',kernel_regularizer=regularizers.l1(1e-4))(inputs)
    layer1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',kernel_regularizer=regularizers.l1(1e-4))(layer0)
    temp = tf.concat([layer0, layer1], axis=1)
    layer2 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',kernel_regularizer=regularizers.l1(1e-4))(temp)
    layer3 = tf.concat([layer0,layer1, layer2], axis=1)
    layer7 = GRU(64, return_sequences=True)(layer3)
    layer7 = Dropout(0.5)(layer7)
    layer7 = layer7+layer3
    layer9 = Flatten()(layer7)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(layer9)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=10, batch_size=100, callbacks = callbacks_list,validation_data=(test_x,test_y))
    x = model.predict(test_x)
    return x