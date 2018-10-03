void setup() {
   surface.setResizable(true);
   selectFolder("写真のあるフォルダを選択してください","loadImages");
}
int indexOfImage = 0;
ArrayList<PImage> originalImages = new ArrayList<PImage>();
ArrayList<String> filename = new ArrayList<String>();
String[] extensions = {
    ".jpg",".gif",".JPG",".png"
};
void loadImages(File selection){
    File[] files = selection.listFiles();
    for(int i = 0; i < files.length; i++){
        for(String extension : extensions){
            if(files[i].getPath().endsWith(extension)){
                PImage originalImage = loadImage(files[i].getAbsolutePath());
                filename.add(files[i].getAbsolutePath());
                println(filename.get(i));
                originalImages.add(originalImage);
            }
        }
    }
    if(!originalImages.isEmpty()){
        PImage img = originalImages.get(0);
        surface.setSize(img.width, img.height);
    }
}

void draw() {
    if(!originalImages.isEmpty()){
        image(originalImages.get(indexOfImage),0,0);
    }
}

void keyPressed(){
   if(keyCode == RIGHT){
       indexOfImage = (indexOfImage+1)%originalImages.size();
       PImage img = originalImages.get(indexOfImage);
       surface.setSize(img.width, img.height);
   }else if(keyCode == LEFT){
       indexOfImage--;
       if(indexOfImage < 0){
           indexOfImage += originalImages.size();
       }
       indexOfImage = (indexOfImage)%originalImages.size();
       PImage img = originalImages.get(indexOfImage);
       surface.setSize(img.width, img.height);
   }
}