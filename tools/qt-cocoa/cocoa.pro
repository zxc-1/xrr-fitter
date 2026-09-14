TEMPLATE = lib
CONFIG += plugin c++17
QT += core-private gui-private
TARGET = qcocoa
QMAKE_MACOSX_DEPLOYMENT_TARGET = 13.0
QMAKE_APPLE_DEVICE_ARCHS = arm64
DESTDIR = $$OUT_PWD/platforms
SOURCES += main.mm qcocoaapplication.mm qcocoaapplicationdelegate.mm qcocoabackingstore.mm qcocoaclipboard.mm qcocoacursor.mm qcocoadrag.mm qcocoaeventdispatcher.mm qcocoahelpers.mm qcocoainputcontext.mm qcocoaintegration.mm qcocoaintrospection.mm qcocoamenu.mm qcocoamenubar.mm qcocoamenuitem.mm qcocoamenuloader.mm qcocoamimetypes.mm qcocoanativeinterface.mm qcocoansmenu.mm qcocoascreen.mm qcocoaservices.mm qcocoasystemtrayicon.mm qcocoatheme.mm qcocoawindow.mm qcocoawindowmanager.mm qiosurfacegraphicsbuffer.mm qmacclipboard.mm qmultitouch_mac.mm qnsview.mm qnswindow.mm qnswindowdelegate.mm qcocoacolordialoghelper.mm qcocoafiledialoghelper.mm qcocoafontdialoghelper.mm qcocoamessagedialog.mm
HEADERS += qcocoaapplication.h qcocoaapplicationdelegate.h qcocoabackingstore.h qcocoaclipboard.h qcocoacursor.h qcocoadrag.h qcocoaeventdispatcher.h qcocoahelpers.h qcocoainputcontext.h qcocoaintegration.h qcocoaintrospection.h qcocoamenu.h qcocoamenubar.h qcocoamenuitem.h qcocoamenuloader.h qcocoamimetypes.h qcocoanativeinterface.h qcocoansmenu.h qcocoascreen.h qcocoaservices.h qcocoasystemtrayicon.h qcocoatheme.h qcocoawindow.h qcocoawindowmanager.h qiosurfacegraphicsbuffer.h qmacclipboard.h qmultitouch_mac_p.h qnsview.h qnswindow.h qnswindowdelegate.h qcocoacolordialoghelper.h qcocoafiledialoghelper.h qcocoafontdialoghelper.h qcocoamessagedialog.h
qtConfig(opengl) {
    SOURCES += qcocoaglcontext.mm
    HEADERS += qcocoaglcontext.h
}
qtConfig(vulkan) {
    SOURCES += qcocoavulkaninstance.mm
    HEADERS += qcocoavulkaninstance.h
}
qtConfig(accessibility) {
    SOURCES += qcocoaaccessibility.mm qcocoaaccessibilityelement.mm
    HEADERS += qcocoaaccessibility.h qcocoaaccessibilityelement.h
}
qtConfig(sessionmanager) {
    SOURCES += qcocoasessionmanager.cpp
    HEADERS += qcocoasessionmanager.h
}
LIBS += -framework Foundation -framework AppKit -framework Carbon -framework CoreServices -framework CoreVideo -framework IOKit -framework IOSurface -framework Metal -framework QuartzCore -framework UniformTypeIdentifiers -framework OpenGL
RESOURCES += qcocoaresources.qrc
