# mad4 readme

## 1. Project Description

---

카이스트 몰입캠프 4주차 프로젝트입니다.

애플 공식 홈페이지를 오마주하여 이미지를 업로드 시 자동으로 물체를 인식하고, nlp모듈을 통한 문구 생성으로 편리하게 중고 거래를 등록할 수 있도록 돕는 플랫폼입니다.

## 2. Image Detection Module

---

이미지 모듈 서버는 yolov4 모델을 활용한 모델 서빙 서버입니다. 이미지 파일을 인풋으로 받아 클래스와 캔버스 좌표값을 반환합니다.

또한 이미지 다중 업로드와 리턴을 지원하는 이미지 업로드 서버로도 작동합니다.

```python
@app.post("/detect")
async def create_file(image: UploadFile = File(...)):
    storage = ""
    imageType = image.content_type
    if imageType == 'image/jpg':
        storage = "data/sample.jpg"
    elif imageType == 'image/png':
        storage = "data/sample.png"
    elif imageType == 'image/jpeg':
        storage = "data/sample.jpeg"
    else:
        return JSONResponse(content={'code':400, 'message':"잘못된 이미지 형식입니다."}, status_code=400)
    with open(storage, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    result = []
    stream = os.popen("./darknet detect cfg/yolov4.cfg cfg/yolov4.weights -ext_output " + storage)
    output = stream.read()
    lst = output.split("\n")
    for i in range(13, len(lst)): 
        if len(lst) > i:
            dic = {'class': '', 'x': 0, 'y': 0, 'w': 0, 'h': 0}
            s = lst[i]
            if len(s) == 0: break
            temp = list(lst[i].split(':'))[0]
            temp = temp.strip()
            if temp not in classes: continue
            dic['class'] = temp
            x_offset = s.rfind('left_x') + len('left_x')
            y_offset = s.rfind('top_y') + len('top_y')
            w_offset = s.rfind('width') + len('width')
            h_offset = s.rfind('height') + len('height')
            dic['x'] = int(s[x_offset+1: x_offset+6])
            dic['y'] = int(s[y_offset+1: y_offset+6])
            dic['w'] = int(s[w_offset+1: w_offset+6])
            dic['h'] = int(s[h_offset+1: h_offset+6])
            result.append(dic)
        else: break
    return JSONResponse(content={'code':200, 'message':"이미지 분석이 완료되었습니다.", 'data':result}, status_code=200)
```

```python
@app.post("/upload_img")
async def save_img(pid: int = Form(...), img_0: UploadFile = File(...), img_1: UploadFile = File(...), img_2: UploadFile = File(...)):
    storage = []
    imgFiles= []

    imgFiles.append(img_0)
    imgFiles.append(img_1)
    imgFiles.append(img_2)

    for i in range(3):
        if imgFiles[i].content_type == 'image/jpg':
            storage.append("images/"+str(pid)+"/"+str(i)+".jpg")
        elif imgFiles[i].content_type == 'image/png':
            storage.append("images/"+str(pid)+"/"+str(i)+".png")
        elif imgFiles[i].content_type == 'image/jpeg':
            storage.append("images/"+str(pid)+"/"+str(i)+".jpeg")
        else:
            return JSONResponse(content={'code':400, 'message':"잘못된 이미지 형식입니다."}, status_code=400)

    directory = "images/"+str(pid)
    for j in range(3):
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(storage[j], "wb") as buffer:
            shutil.copyfileobj(imgFiles[j].file, buffer)

    return JSONResponse(content={'code':200, 'message':"이미지 저장이 완료되었습니다."}, status_code=200)
```

## 3. NLP Module

---

NLP모듈은 중고품에서 흔히 나타나는 특성들인 ‘내구성’, ‘사용감’, ‘디자인’ 요소들을 기준으로 훈련받을 모델을 서빙하며 KoGPT2 모델을 활용합니다. 애플 공식 홈페이지와 비슷한 말투를 훈련 데이터 셋으로 두고, 이에 대해 data augmentation과정을 거쳐 내구성, 사용감 등의 키워드를 받았을 때 자연스러운 애플 공식 홈페이지 스타일의 말투를 반환합니다.

```python
def idea_maker(self, category):
        sent='0'
        tok = self.tokenizer
        sent_tokens = tok.tokenize(sent)
        with torch.no_grad():
            a = ''
            while 1:
                input_ids = torch.LongTensor(tok.encode(U_TKN + category + SENT + sent + S_TKN + a)).unsqueeze(dim=0).cuda()
                pred = self(input_ids).cuda()
                gen = tok.convert_ids_to_tokens(
                    torch.argmax(
                        pred,
                        dim=-1).squeeze().cpu().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace('▁', ' ')

            a.replace(' ', '')
            return a

    def nbest_ideas_maker(self, category):
        # input_ids = self.tokenizer.encode(category, return_tensors='tf')
        input_ids = torch.LongTensor(self.tokenizer.encode(BOS + category + EOS)).unsqueeze(dim=0).cuda()
        beam_outputs = self.model.generate(
            input_ids, 
            max_length=128, 
            num_beams=10, 
            no_repeat_ngram_size=2, 
            num_return_sequences=10,
            early_stopping=True
        )

        print("Output:nbest\n" + 100 * '-')
        for i, beam_output in enumerate(beam_outputs):
            print("{}: {}".format(i, self.tokenizer.decode(beam_output, skip_special_tokens=True)))        
        # result = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)

        result = []
        for i in range(0, 10):
            a = self.tokenizer.batch_decode(beam_outputs.tolist(), skip_special_tokens=True)[i]
            idea = a.replace(category+' ', '')
            result.append(idea)
        return result

    def temperature_idea_maker(self, category):
        input_ids = torch.LongTensor(self.tokenizer.encode(BOS + category + EOS)).unsqueeze(dim=0).cuda()
        temp_outputs = self.model.generate(
            input_ids,
            do_sample=True,
            max_length = 128,
            top_k=0,
            temperature=0.8
        )
        result = self.tokenizer.decode(temp_outputs[0], skip_special_tokens = True)
        final_result = result.replace(category+' ', '')
        print("Output:temperature\n" + 100 * '-')
        print(final_result)

        return final_result
```

## 4. Backend Server

---

백엔드 서버는 유저와 거래, 결제 기능들을 지원합니다. JWT를 활용한 로그인, 회원가입 및 토큰 발급의 기본적인 기능부터  product, user, trade등에 대한 CRUD기능 및 포인트 충전을 위한 결제 등을 지원합니다. 거래는 iamport모듈을 활용한 포인트 충전과 이를 기반으로 한 예치식 안전결제 시스템을 채용했습니다. 

Java Springboot와 JPA, MySQL를 활용하여 구성했습니다.

```java
public DefaultResponse startTrade(Forms.PreferForm preferForm){
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication == null) {
            return DefaultResponse.res(StatusCode.BAD_REQUEST, ResponseMessage.TOKEN_FAILED);
        }
        if (authentication.getName().equals("anonymousUser")){
            return DefaultResponse.res(StatusCode.NEED_REFRESH, ResponseMessage.REQUIRES_TOKEN_UPDATE);
        }
        Optional<Users> users = usersRepository.findById(authentication.getName());
        if (!users.isPresent()) {
            return DefaultResponse.res(StatusCode.NOT_FOUND, ResponseMessage.NOT_FOUND_USER);
        }
        Users seller = users.get();
        Optional<Users> users2 = usersRepository.findById(preferForm.getUid());
        if (!users2.isPresent()){
            return DefaultResponse.res(StatusCode.NOT_FOUND, ResponseMessage.NOT_FOUND_USER);
        }
        Users buyer = users2.get();
        Optional<Products> products = productsRepository.findById(preferForm.getProduct_id());
        if (!products.isPresent()){
            return DefaultResponse.res(StatusCode.NOT_FOUND, ResponseMessage.NOT_FOUND_PRODUCT);
        }
        Products product = products.get();
        List<Trade> trades = tradeRepository.checkTradeExists(seller.getUid(), buyer.getUid(), product.getPid());
        if (trades.size() != 0){
            return DefaultResponse.res(StatusCode.BAD_REQUEST, ResponseMessage.TRADE_ALREADY_EXISTS);
        }
        Trade trade = Trade.builder()
                .buyer(buyer.getUid())
                .seller(seller.getUid())
                .product_id(product.getPid())
                .bill(product.getPrice())
                .completion(0)
                .buyer_confirm(0)
                .seller_confirm(0)
                .build();
        tradeRepository.save(trade);
        buyer.setPoints(-product.getPrice());
        usersRepository.save(buyer);
        return DefaultResponse.res(StatusCode.OK, ResponseMessage.TRADE_REGISTERED, trade);
    }

    public DefaultResponse tradeList(){
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication == null) {
            return DefaultResponse.res(StatusCode.BAD_REQUEST, ResponseMessage.TOKEN_FAILED);
        }
        if (authentication.getName().equals("anonymousUser")){
            return DefaultResponse.res(StatusCode.NEED_REFRESH, ResponseMessage.REQUIRES_TOKEN_UPDATE);
        }
        Optional<Users> users = usersRepository.findById(authentication.getName());
        if (!users.isPresent()) {
            return DefaultResponse.res(StatusCode.NOT_FOUND, ResponseMessage.NOT_FOUND_USER);
        }
        Users mySelf = users.get();
        List<Trade> trades = tradeRepository.getTradeList(mySelf.getUid());
        return DefaultResponse.res(StatusCode.OK, ResponseMessage.READ_USER, trades);
    }
```

## 5. Frontend  WebPage

---

웹 페이지는 Vue.js를 기반으로 생성되었으며, 스크롤 이벤트 등의 다양한 사용자 인터렉션을 제공합니다. NLP모듈과 마찬가지로 애플 공식 홈페이지를 오마주하였으며 유저 페이지, 상품 등록, 상품 거래 등의 기능을 지원합니다.

```jsx
const routes = [
  {
    path: "/",
    name: "Home",
    component: Home,
  },
  {
    path: "/about",
    name: "About",
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () =>
      import(/* webpackChunkName: "about" */ "../views/About.vue"),
  },
  {
    path: "/learn",
    name: "Learn",
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () =>
      import(/* webpackChunkName: "about" */ "../views/Learn.vue"),
  },
  {
    path: "/ImageSubmit",
    name: "ImageSubmit",
    component: () =>
      import(/* webpackChunkName: "about" */ "../views/ImageSubmit.vue"),
  },
  {
    path: "/MakeArticle",
    name: "MakeArticle",
    component: () =>
      import(/* webpackChunkName: "about" */ "../views/MakeArticle.vue"),
  },
  {
    path: "/MultiImg",
    name: "MultiImg",
    component: () =>
      import(/* webpackChunkName: "about" */ "../views/MultiImg.vue"),
  },
  {
    path: "/Test",
    name: "Test",
    component: () =>
      import(/* webpackChunkName: "about" */ "../views/Test.vue"),
  },
  {
    path: "/detail",
    name: "Detail",
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () =>
      import(/* webpackChunkName: "about" */ "../views/Detail.vue"),
  },
   {
    path: "/price",
    name: "Price",
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () =>
      import(/* webpackChunkName: "about" */ "../views/Price.vue"),
  },
  {
    path: "/Signup",
    name: "Signup",
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () =>
      import(/* webpackChunkName: "about" */ "../views/Signup.vue"),
  },
  {
    path: "/Charge",
    name: "Charge",
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () =>
      import(/* webpackChunkName: "about" */ "../views/Charge.vue"),
  },
];

const router = createRouter({
  history: createWebHashHistory(),
  routes,
});

export default router;
```